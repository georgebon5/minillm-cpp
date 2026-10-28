#include "minillm/model.hpp"
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

} // namespace minillm
