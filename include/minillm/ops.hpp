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

// ── Attention ─────────────────────────────────────────────────────────────────

// Numerically stable softmax over a single row of `n` floats, in-place.
// Exported so the sampler can reuse it for logit processing.
void softmax_inplace(float* row, size_t n);

// Scaled dot-product attention for a single head.
//   q, k, v : [seq_len, head_dim]  F32
//   causal  : if true, mask out j > i (upper triangle set to -inf before softmax)
//   returns : [seq_len, head_dim]  F32  (owning)
Tensor scaled_dot_product_attention(const Tensor& q,
                                     const Tensor& k,
                                     const Tensor& v,
                                     bool causal = false);

// ── Elementwise ops ───────────────────────────────────────────────────────────

// a[i] += b[i]  in-place.  Both tensors must be F32 with the same shape.
void add_inplace(Tensor& a, const Tensor& b);

// ── MLP ───────────────────────────────────────────────────────────────────────

// SiLU activation: silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
inline float silu(float x) { return x / (1.f + std::exp(-x)); }

// SwiGLU MLP: out = W_down * (silu(W_gate * x) ⊙ W_up * x)
//   x      : [seq_len, hidden_size]          F32
//   W_gate : [intermediate_size, hidden_size] F32
//   W_up   : [intermediate_size, hidden_size] F32
//   W_down : [hidden_size, intermediate_size] F32
//   returns: [seq_len, hidden_size]            F32  (owning)
Tensor swiglu_mlp(const Tensor& x,
                   const Tensor& W_gate,
                   const Tensor& W_up,
                   const Tensor& W_down);

// ── Grouped-Query Attention (GQA) layer — full multi-head projection + attention + output projection.
//
//   x          : [seq_len, hidden_size]            F32
//   Wq         : [n_heads     * head_dim, hidden]  F32
//   Wk         : [n_kv_heads  * head_dim, hidden]  F32
//   Wv         : [n_kv_heads  * head_dim, hidden]  F32
//   Wo         : [hidden,  n_heads * head_dim]      F32
//   rope       : precomputed cos/sin tables
//   n_heads    : number of query heads
//   n_kv_heads : number of key/value heads (n_heads / n_kv_heads = kv_groups)
//   pos_offset : sequence position of the first token in x (0 for prefill)
//   returns    : [seq_len, hidden_size]  F32  (owning)
Tensor gqa_attention(const Tensor& x,
                     const Tensor& Wq,
                     const Tensor& Wk,
                     const Tensor& Wv,
                     const Tensor& Wo,
                     const RopeCache& rope,
                     int n_heads,
                     int n_kv_heads,
                     int pos_offset = 0);

} // namespace minillm
