#include "minillm/model.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

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

// ── Quantized-weight simulation ──────────────────────────────────────────────

ModelWeights quantize_weights(const ModelWeights& src, DType qdtype) {
    auto sim = [qdtype](const Tensor& t) {
        return dequantize(quantize(t, qdtype));
    };

    ModelWeights w;
    w.embed      = src.embed;       // kept F32 (embedding is a lookup, not linear)
    w.final_norm = src.final_norm;  // kept F32 (elementwise scale, tiny)

    w.layers.resize(src.layers.size());
    for (size_t l = 0; l < src.layers.size(); ++l) {
        const LayerWeights& s = src.layers[l];
        LayerWeights&       d = w.layers[l];
        d.attn_norm = s.attn_norm;  // RMSNorm scale, not a projection
        d.mlp_norm  = s.mlp_norm;
        d.Wq     = sim(s.Wq);
        d.Wk     = sim(s.Wk);
        d.Wv     = sim(s.Wv);
        d.Wo     = sim(s.Wo);
        d.W_gate = sim(s.W_gate);
        d.W_up   = sim(s.W_up);
        d.W_down = sim(s.W_down);
    }
    w.lm_head = sim(src.lm_head);
    return w;
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

// Apply repetition penalty: for each recently-seen token t,
//   logits[t] /= penalty   (if logits[t] > 0)
//   logits[t] *= penalty   (if logits[t] <= 0)
// Discourages re-emitting tokens that appeared in the last `window` positions.
// Writes into `scratch` so the caller's logit buffer stays untouched.
static void apply_rep_penalty(const float* logits, size_t vocab,
                              const std::vector<int32_t>& history,
                              const SamplingParams& sp,
                              std::vector<float>& scratch) {
    scratch.assign(logits, logits + vocab);
    if (sp.rep_penalty <= 1.0f || sp.rep_window <= 0) return;

    const size_t start = history.size() > static_cast<size_t>(sp.rep_window)
                         ? history.size() - static_cast<size_t>(sp.rep_window)
                         : 0;
    for (size_t i = start; i < history.size(); ++i) {
        int32_t t = history[i];
        if (t < 0 || static_cast<size_t>(t) >= vocab) continue;
        scratch[t] = (scratch[t] > 0.f) ? scratch[t] / sp.rep_penalty
                                        : scratch[t] * sp.rep_penalty;
    }
}

// Pick the next token given raw logits + sampling params.
// temperature == 0 → greedy argmax; otherwise top-K softmax sampling.
static int32_t sample_next(const float* logits, size_t vocab,
                           const SamplingParams& sp, std::mt19937& rng,
                           const std::vector<int32_t>& history,
                           std::vector<float>& scratch) {
    apply_rep_penalty(logits, vocab, history, sp, scratch);
    const float* lp = scratch.data();

    if (sp.temperature <= 0.0f) {
        int32_t best = 0; float best_val = lp[0];
        for (size_t i = 1; i < vocab; ++i)
            if (lp[i] > best_val) { best_val = lp[i]; best = static_cast<int32_t>(i); }
        return best;
    }

    const size_t k   = std::min<size_t>(static_cast<size_t>(std::max(1, sp.top_k)), vocab);
    const float  inv = 1.0f / sp.temperature;

    std::vector<std::pair<float, int32_t>> scored(vocab);
    for (size_t i = 0; i < vocab; ++i)
        scored[i] = { lp[i] * inv, static_cast<int32_t>(i) };

    std::partial_sort(scored.begin(), scored.begin() + k, scored.end(),
                      [](auto& a, auto& b){ return a.first > b.first; });

    // Numerically stable softmax over the top-K.
    float max_v = scored[0].first;
    std::vector<float> probs(k);
    float sum = 0.f;
    for (size_t i = 0; i < k; ++i) {
        probs[i] = std::exp(scored[i].first - max_v);
        sum += probs[i];
    }
    for (auto& p : probs) p /= sum;

    // Nucleus (top-p) truncation: keep smallest set whose cumulative prob >= top_p.
    size_t nucleus = k;
    if (sp.top_p < 1.0f) {
        float cum = 0.f;
        for (size_t i = 0; i < k; ++i) {
            cum += probs[i];
            if (cum >= sp.top_p) { nucleus = i + 1; break; }
        }
        // Renormalize the kept slice.
        float s = 0.f;
        for (size_t i = 0; i < nucleus; ++i) s += probs[i];
        for (size_t i = 0; i < nucleus; ++i) probs[i] /= s;
    }

    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    float r = u(rng), acc = 0.f;
    for (size_t i = 0; i < nucleus; ++i) {
        acc += probs[i];
        if (r < acc) return scored[i].second;
    }
    return scored[nucleus - 1].second;
}

std::vector<int32_t> generate(const ModelWeights&       w,
                               const ModelConfig&         cfg,
                               const RopeCache&           rope,
                               std::span<const int32_t>   prompt_ids,
                               int                        max_new_tokens,
                               int32_t                    eos_id,
                               const SamplingParams&      sampling) {
    KVCache kv = make_kvcache(cfg);

    Tensor logits;
    for (int32_t id : prompt_ids)
        logits = forward_one(w, cfg, rope, id, kv);

    const size_t vocab = static_cast<size_t>(cfg.vocab_size);
    std::vector<int32_t> generated;
    generated.reserve(static_cast<size_t>(max_new_tokens));

    std::mt19937 rng(sampling.seed ? sampling.seed : std::random_device{}());

    // History includes prompt + already-generated for repetition penalty.
    std::vector<int32_t> history(prompt_ids.begin(), prompt_ids.end());
    std::vector<float>   scratch;
    scratch.reserve(vocab);

    for (int step = 0; step < max_new_tokens; ++step) {
        int32_t next_id = sample_next(logits.data_as<float>(), vocab,
                                       sampling, rng, history, scratch);
        generated.push_back(next_id);
        history.push_back(next_id);
        if (next_id == eos_id || (sampling.stop2 >= 0 && next_id == sampling.stop2)) break;
        logits = forward_one(w, cfg, rope, next_id, kv);
    }
    return generated;
}

// ── Direct BF16→INT8 weight loading ──────────────────────────────────────────

ModelWeightsQ load_weights_q(const SafetensorsFile& sf, const ModelConfig& cfg, DType qdtype) {
    ModelWeightsQ w;
    w.embed = sf.tensor_f32("model.embed_tokens.weight");
    w.layers.resize(static_cast<size_t>(cfg.num_hidden_layers));
    for (int l = 0; l < cfg.num_hidden_layers; ++l) {
        LayerWeightsQ& lw = w.layers[static_cast<size_t>(l)];
        const std::string p = "model.layers." + std::to_string(l) + ".";
        lw.attn_norm = sf.tensor_f32(p + "input_layernorm.weight");
        lw.Wq        = sf.tensor_q(p + "self_attn.q_proj.weight",  qdtype);
        lw.Wk        = sf.tensor_q(p + "self_attn.k_proj.weight",  qdtype);
        lw.Wv        = sf.tensor_q(p + "self_attn.v_proj.weight",  qdtype);
        lw.Wo        = sf.tensor_q(p + "self_attn.o_proj.weight",  qdtype);
        lw.mlp_norm  = sf.tensor_f32(p + "post_attention_layernorm.weight");
        lw.W_gate    = sf.tensor_q(p + "mlp.gate_proj.weight",     qdtype);
        lw.W_up      = sf.tensor_q(p + "mlp.up_proj.weight",       qdtype);
        lw.W_down    = sf.tensor_q(p + "mlp.down_proj.weight",     qdtype);
    }
    w.final_norm = sf.tensor_f32("model.norm.weight");
    const std::string lm_name = sf.has("lm_head.weight") ? "lm_head.weight"
                                                          : "model.embed_tokens.weight";
    w.lm_head = sf.tensor_q(lm_name, qdtype);
    return w;
}

// ── Quantized forward helpers (INT8/INT4 native) ──────────────────────────────

static Tensor swiglu_mlp_q(const Tensor&      x,
                             const QuantTensor& W_gate,
                             const QuantTensor& W_up,
                             const QuantTensor& W_down) {
    Tensor g = linear_q(x, W_gate);
    Tensor u = linear_q(x, W_up);
    float*       gp = g.data_as<float>();
    const float* up = u.data_as<float>();
    for (size_t i = 0, n = g.numel(); i < n; ++i)
        gp[i] = silu(gp[i]) * up[i];
    return linear_q(g, W_down);
}

static Tensor gqa_attn_cached_q(const Tensor&      x,
                                  const QuantTensor& Wq,
                                  const QuantTensor& Wk,
                                  const QuantTensor& Wv,
                                  const QuantTensor& Wo,
                                  const RopeCache&   rope,
                                  KVCache::Layer&    kv_layer,
                                  int n_heads,
                                  int n_kv_heads,
                                  int pos) {
    assert(x.shape()[0] == 1);
    const size_t head_dim  = Wq.data.shape()[0] / static_cast<size_t>(n_heads);
    const int    kv_groups = n_heads / n_kv_heads;

    Tensor Q     = linear_q(x, Wq);
    Tensor K_new = linear_q(x, Wk);
    Tensor V_new = linear_q(x, Wv);

    rope_apply(Q,     rope, pos);
    rope_apply(K_new, rope, pos);

    kv_layer.append(K_new.data_as<float>(), V_new.data_as<float>());
    const size_t seq = kv_layer.seq_len;

    const size_t n_h = static_cast<size_t>(n_heads);
    Tensor attn_out({1, n_h * head_dim}, DType::F32);
    float* out_p = attn_out.data_as<float>();

    for (int h = 0; h < n_heads; ++h) {
        const size_t kv_h = static_cast<size_t>(h / kv_groups);
        Tensor q_h = slice_cols_raw(Q.data_as<float>(), 1, n_h * head_dim,
                                    static_cast<size_t>(h) * head_dim, head_dim);
        Tensor k_h = slice_cols_raw(kv_layer.k.data(), seq, kv_layer.kv_dim,
                                    kv_h * head_dim, head_dim);
        Tensor v_h = slice_cols_raw(kv_layer.v.data(), seq, kv_layer.kv_dim,
                                    kv_h * head_dim, head_dim);
        Tensor out_h = scaled_dot_product_attention(q_h, k_h, v_h, false);
        std::memcpy(out_p + static_cast<size_t>(h) * head_dim,
                    out_h.data_as<float>(), head_dim * sizeof(float));
    }
    return linear_q(attn_out, Wo);
}

Tensor forward_one_q(const ModelWeightsQ& w,
                     const ModelConfig&   cfg,
                     const RopeCache&     rope,
                     int32_t              token_id,
                     KVCache&             kv) {
    const int pos = static_cast<int>(kv.layers[0].seq_len);
    std::array<int32_t, 1> ids_arr = {token_id};
    Tensor x = embedding_lookup(w.embed, ids_arr);

    for (int l = 0; l < cfg.num_hidden_layers; ++l) {
        const LayerWeightsQ& lw = w.layers[static_cast<size_t>(l)];
        Tensor normed   = rms_norm(x, lw.attn_norm, cfg.rms_norm_eps);
        Tensor attn_out = gqa_attn_cached_q(normed,
                                             lw.Wq, lw.Wk, lw.Wv, lw.Wo,
                                             rope,
                                             kv.layers[static_cast<size_t>(l)],
                                             cfg.num_attention_heads,
                                             cfg.num_key_value_heads,
                                             pos);
        add_inplace(x, attn_out);
        normed         = rms_norm(x, lw.mlp_norm, cfg.rms_norm_eps);
        Tensor mlp_out = swiglu_mlp_q(normed, lw.W_gate, lw.W_up, lw.W_down);
        add_inplace(x, mlp_out);
    }
    x = rms_norm(x, w.final_norm, cfg.rms_norm_eps);
    return linear_q(x, w.lm_head);
}

std::vector<int32_t> generate_q(const ModelWeightsQ&     w,
                                  const ModelConfig&        cfg,
                                  const RopeCache&          rope,
                                  std::span<const int32_t>  prompt_ids,
                                  int                       max_new_tokens,
                                  int32_t                   eos_id,
                                  const SamplingParams&     sampling) {
    KVCache kv = make_kvcache(cfg);
    Tensor logits;
    for (int32_t id : prompt_ids)
        logits = forward_one_q(w, cfg, rope, id, kv);

    const size_t vocab = static_cast<size_t>(cfg.vocab_size);
    std::vector<int32_t> generated;
    generated.reserve(static_cast<size_t>(max_new_tokens));

    std::mt19937 rng(sampling.seed ? sampling.seed : std::random_device{}());
    std::vector<int32_t> history(prompt_ids.begin(), prompt_ids.end());
    std::vector<float>   scratch;
    scratch.reserve(vocab);

    for (int step = 0; step < max_new_tokens; ++step) {
        int32_t next_id = sample_next(logits.data_as<float>(), vocab,
                                       sampling, rng, history, scratch);
        generated.push_back(next_id);
        history.push_back(next_id);
        if (next_id == eos_id || (sampling.stop2 >= 0 && next_id == sampling.stop2)) break;
        logits = forward_one_q(w, cfg, rope, next_id, kv);
    }
    return generated;
}

} // namespace minillm
