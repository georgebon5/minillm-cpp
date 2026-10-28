#pragma once
#include <span>
#include <string>
#include <vector>
#include "model_config.hpp"
#include "ops.hpp"
#include "safetensors.hpp"
#include "tensor.hpp"

namespace minillm {

// Weights for a single transformer block.
struct LayerWeights {
    Tensor attn_norm;   // [hidden]                    pre-attention RMSNorm
    Tensor Wq;          // [n_heads     * head_dim, hidden]
    Tensor Wk;          // [n_kv_heads  * head_dim, hidden]
    Tensor Wv;          // [n_kv_heads  * head_dim, hidden]
    Tensor Wo;          // [hidden, n_heads * head_dim]
    Tensor mlp_norm;    // [hidden]                    pre-MLP RMSNorm
    Tensor W_gate;      // [intermediate, hidden]
    Tensor W_up;        // [intermediate, hidden]
    Tensor W_down;      // [hidden, intermediate]
};

// All model weights loaded into RAM as F32.
struct ModelWeights {
    Tensor embed;                      // [vocab, hidden]
    std::vector<LayerWeights> layers;  // one per transformer block
    Tensor final_norm;                 // [hidden]
    Tensor lm_head;                    // [vocab, hidden]  (may equal embed if tied)
};

// Load all weights from a safetensors file (BF16 → F32 conversion included).
ModelWeights load_weights(const SafetensorsFile& sf, const ModelConfig& cfg);

// Full causal forward pass.
//   ids     : token IDs for the input sequence
//   rope    : precomputed RoPE tables (build once, reuse)
//   returns : [seq_len, vocab_size]  F32 logits (all positions)
Tensor forward(const ModelWeights& weights,
               const ModelConfig&  cfg,
               const RopeCache&    rope,
               std::span<const int32_t> ids);

} // namespace minillm
