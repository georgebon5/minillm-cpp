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

} // namespace minillm
