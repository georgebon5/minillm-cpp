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

} // namespace minillm
