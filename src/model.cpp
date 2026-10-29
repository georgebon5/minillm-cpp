#include "minillm/model.hpp"
#include <cassert>
#include <cstring>
#include <stdexcept>
#include <string>

namespace minillm {

ModelWeights load_weights(const SafetensorsFile& sf, const ModelConfig& cfg) {
    ModelWeights w;

    w.embed = sf.tensor_f32("model.embed_tokens.weight");

    w.layers.resize(static_cast<size_t>(cfg.num_hidden_layers));
    for (int l = 0; l < cfg.num_hidden_layers; ++l) {
        LayerWeights& lw   = w.layers[static_cast<size_t>(l)];
        const std::string p = "model.layers." + std::to_string(l) + ".";

        lw.attn_norm = sf.tensor_f32(p + "input_layernorm.weight");
        lw.Wq        = sf.tensor_f32(p + "self_attn.q_proj.weight");
        lw.Wk        = sf.tensor_f32(p + "self_attn.k_proj.weight");
        lw.Wv        = sf.tensor_f32(p + "self_attn.v_proj.weight");
        lw.Wo        = sf.tensor_f32(p + "self_attn.o_proj.weight");
        lw.mlp_norm  = sf.tensor_f32(p + "post_attention_layernorm.weight");
        lw.W_gate    = sf.tensor_f32(p + "mlp.gate_proj.weight");
        lw.W_up      = sf.tensor_f32(p + "mlp.up_proj.weight");
        lw.W_down    = sf.tensor_f32(p + "mlp.down_proj.weight");
    }

    w.final_norm = sf.tensor_f32("model.norm.weight");

    // lm_head may be stored separately or tied to the embedding table.
    w.lm_head = sf.has("lm_head.weight")
                    ? sf.tensor_f32("lm_head.weight")
                    : sf.tensor_f32("model.embed_tokens.weight");

    return w;
}

Tensor forward(const ModelWeights& w,
               const ModelConfig&  cfg,
               const RopeCache&    rope,
               std::span<const int32_t> ids) {
    // ── Embedding lookup ─────────────────────────────────────────────────────
    Tensor x = embedding_lookup(w.embed, ids);  // [seq, hidden]

    // ── Transformer blocks ───────────────────────────────────────────────────
    for (int l = 0; l < cfg.num_hidden_layers; ++l) {
        const LayerWeights& lw = w.layers[static_cast<size_t>(l)];

        // Pre-attention RMSNorm + attention + residual
        Tensor normed   = rms_norm(x, lw.attn_norm, cfg.rms_norm_eps);
        Tensor attn_out = gqa_attention(normed,
                                        lw.Wq, lw.Wk, lw.Wv, lw.Wo,
                                        rope,
                                        cfg.num_attention_heads,
                                        cfg.num_key_value_heads,
                                        /*pos_offset=*/0);
        add_inplace(x, attn_out);

        // Pre-MLP RMSNorm + MLP + residual
        normed          = rms_norm(x, lw.mlp_norm, cfg.rms_norm_eps);
        Tensor mlp_out  = swiglu_mlp(normed, lw.W_gate, lw.W_up, lw.W_down);
        add_inplace(x, mlp_out);
    }

    // ── Final norm + LM head ─────────────────────────────────────────────────
    x = rms_norm(x, w.final_norm, cfg.rms_norm_eps);
    return linear(x, w.lm_head);   // [seq, vocab]
}

// ── KV-cache helpers ─────────────────────────────────────────────────────────

KVCache make_kvcache(const ModelConfig& cfg) {
    KVCache kv;
    kv.layers.resize(static_cast<size_t>(cfg.num_hidden_layers));
    const size_t kv_dim = static_cast<size_t>(cfg.num_key_value_heads) *
                          static_cast<size_t>(cfg.head_dim);
    for (auto& l : kv.layers) l.kv_dim = kv_dim;
    return kv;
}

// Extract column slice [col_start, col_start+width) from a raw row-major
// [rows, n_cols] float array → new owning [rows, width] Tensor.
static Tensor slice_cols_raw(const float* data, size_t rows, size_t n_cols,
                              size_t col_start, size_t width) {
    Tensor out({rows, width}, DType::F32);
    float* dst = out.data_as<float>();
    for (size_t r = 0; r < rows; ++r)
        std::memcpy(dst + r * width,
                    data + r * n_cols + col_start,
                    width * sizeof(float));
    return out;
}

// GQA attention for a single new token against the full KV cache.
// Projects x [1, hidden] → Q/K/V, appends K/V to kv_layer, returns [1, hidden].
static Tensor gqa_attn_cached(const Tensor&    x,
                               const Tensor&    Wq,
                               const Tensor&    Wk,
                               const Tensor&    Wv,
                               const Tensor&    Wo,
                               const RopeCache& rope,
                               KVCache::Layer&  kv_layer,
                               int n_heads,
                               int n_kv_heads,
                               int pos) {
    assert(x.shape()[0] == 1);
    const size_t head_dim  = Wq.shape()[0] / static_cast<size_t>(n_heads);
    const int    kv_groups = n_heads / n_kv_heads;

    // 1. Project
    Tensor Q     = linear(x, Wq);  // [1, n_heads    * head_dim]
    Tensor K_new = linear(x, Wk);  // [1, n_kv_heads * head_dim]
    Tensor V_new = linear(x, Wv);  // [1, n_kv_heads * head_dim]

    // 2. RoPE at current position
    rope_apply(Q,     rope, pos);
    rope_apply(K_new, rope, pos);

    // 3. Append to cache
    kv_layer.append(K_new.data_as<float>(), V_new.data_as<float>());
    const size_t seq = kv_layer.seq_len;  // now includes the new token

    // 4. Per-head attention against full cache (no causal mask needed: single query)
    const size_t n_h = static_cast<size_t>(n_heads);
    Tensor attn_out({1, n_h * head_dim}, DType::F32);
    float* out_p = attn_out.data_as<float>();

    for (int h = 0; h < n_heads; ++h) {
        const size_t kv_h = static_cast<size_t>(h / kv_groups);

        Tensor q_h = slice_cols_raw(Q.data_as<float>(), 1,
                                    static_cast<size_t>(n_heads) * head_dim,
                                    static_cast<size_t>(h) * head_dim, head_dim);
        Tensor k_h = slice_cols_raw(kv_layer.k.data(), seq, kv_layer.kv_dim,
                                    kv_h * head_dim, head_dim);
        Tensor v_h = slice_cols_raw(kv_layer.v.data(), seq, kv_layer.kv_dim,
                                    kv_h * head_dim, head_dim);

        Tensor out_h = scaled_dot_product_attention(q_h, k_h, v_h, /*causal=*/false);

        const float* src = out_h.data_as<float>();
        std::memcpy(out_p + static_cast<size_t>(h) * head_dim,
                    src,
                    head_dim * sizeof(float));
    }

    // 5. Output projection
    return linear(attn_out, Wo);
}

Tensor forward_one(const ModelWeights& w,
                   const ModelConfig&  cfg,
                   const RopeCache&    rope,
                   int32_t             token_id,
                   KVCache&            kv) {
    const int pos = static_cast<int>(kv.layers[0].seq_len);

    // Embed single token
    std::array<int32_t, 1> ids_arr = {token_id};
    Tensor x = embedding_lookup(w.embed, ids_arr);   // [1, hidden]

    for (int l = 0; l < cfg.num_hidden_layers; ++l) {
        const LayerWeights& lw = w.layers[static_cast<size_t>(l)];

        Tensor normed   = rms_norm(x, lw.attn_norm, cfg.rms_norm_eps);
        Tensor attn_out = gqa_attn_cached(normed,
                                          lw.Wq, lw.Wk, lw.Wv, lw.Wo,
                                          rope,
                                          kv.layers[static_cast<size_t>(l)],
                                          cfg.num_attention_heads,
                                          cfg.num_key_value_heads,
                                          pos);
        add_inplace(x, attn_out);

        normed         = rms_norm(x, lw.mlp_norm, cfg.rms_norm_eps);
        Tensor mlp_out = swiglu_mlp(normed, lw.W_gate, lw.W_up, lw.W_down);
        add_inplace(x, mlp_out);
    }

    x = rms_norm(x, w.final_norm, cfg.rms_norm_eps);
    return linear(x, w.lm_head);   // [1, vocab]
}

// ── Greedy generation loop ────────────────────────────────────────────────────

std::vector<int32_t> generate(const ModelWeights&       w,
                               const ModelConfig&         cfg,
                               const RopeCache&           rope,
                               std::span<const int32_t>   prompt_ids,
                               int                        max_new_tokens,
                               int32_t                    eos_id) {
    KVCache kv = make_kvcache(cfg);

    // Prefill: run every prompt token through forward_one to populate the KV
    // cache; keep only the last token's logits to seed the decode loop.
    Tensor logits;
    for (int32_t id : prompt_ids)
        logits = forward_one(w, cfg, rope, id, kv);

    // Decode: argmax the current logits, emit the token, then step forward.
    const size_t vocab = static_cast<size_t>(cfg.vocab_size);
    std::vector<int32_t> generated;
    generated.reserve(static_cast<size_t>(max_new_tokens));

    for (int step = 0; step < max_new_tokens; ++step) {
        const float* lp = logits.data_as<float>();

        int32_t next_id = 0;
        float   best    = lp[0];
        for (size_t v = 1; v < vocab; ++v)
            if (lp[v] > best) { best = lp[v]; next_id = static_cast<int32_t>(v); }

        generated.push_back(next_id);
        if (next_id == eos_id) break;

        logits = forward_one(w, cfg, rope, next_id, kv);
    }

    return generated;
}

} // namespace minillm
