#pragma once
#include <span>
#include "tensor.hpp"

namespace minillm {

// ── Embedding ─────────────────────────────────────────────────────────────────

// Fetch token embedding vectors from a weight matrix.
//   weight : [vocab_size, hidden_size]  F32
//   ids    : sequence of token IDs, each in [0, vocab_size)
//   returns: [seq_len, hidden_size]     F32  (owning)
Tensor embedding_lookup(const Tensor& weight,
                         std::span<const int32_t> ids);

// ── Normalization ─────────────────────────────────────────────────────────────

// Root-Mean-Square Layer Normalization (out-of-place).
//   x      : [seq_len, hidden] or [hidden]   F32
//   weight : [hidden]                         F32  (learned scale, γ)
//   eps    : small constant for numerical stability (typically 1e-5)
//   returns: same shape as x                  F32  (owning)
Tensor rms_norm(const Tensor& x, const Tensor& weight, float eps);

// ── Linear projection ─────────────────────────────────────────────────────────

// Computes y = x @ W^T + b  (bias is optional).
//   x : [seq_len, in_features]      F32
//   W : [out_features, in_features]  F32  (rows = output neurons)
//   b : [out_features]               F32  (pass nullptr for no bias)
//   returns: [seq_len, out_features]  F32  (owning)
Tensor linear(const Tensor& x, const Tensor& W, const Tensor* b = nullptr);

// ── RoPE (Rotary Position Embedding) ─────────────────────────────────────────

// Precomputed cos/sin tables for RoPE.
// Shape of each table: [max_position, head_dim/2]
struct RopeCache {
    Tensor cos;  // cos(m * θ_i)
    Tensor sin;  // sin(m * θ_i)
};

// Precompute RoPE cos/sin tables for all positions 0..max_pos-1.
//   θ_i = 1 / (rope_theta ^ (2i / head_dim))  for i in [0, head_dim/2)
//
//   max_pos    : typically model.max_position_embeddings (e.g. 2048)
//   head_dim   : hidden_size / num_attention_heads (e.g. 64)
//   rope_theta : frequency base (e.g. 10000.0)
RopeCache build_rope_cache(int max_pos, int head_dim, float rope_theta);

// Apply RoPE rotation in-place to a projected Q or K tensor.
//   x          : [seq_len, n_heads * head_dim]  F32  (modified in-place)
//   cache      : precomputed tables; head_dim inferred as 2 * cache.cos.shape()[1]
//   pos_offset : starting sequence position (0 for prefill, current pos for decode step)
void rope_apply(Tensor& x, const RopeCache& cache, int pos_offset = 0);

} // namespace minillm
