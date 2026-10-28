#include "minillm/ops.hpp"
#include <cassert>
#include <cstring>
#include <stdexcept>

namespace minillm {

// ── Embedding ─────────────────────────────────────────────────────────────────

Tensor embedding_lookup(const Tensor& weight, std::span<const int32_t> ids) {
    assert(weight.ndim() == 2 && weight.dtype() == DType::F32);

    const size_t vocab_size  = weight.shape()[0];
    const size_t hidden_size = weight.shape()[1];
    const size_t seq_len     = ids.size();

    Tensor out({seq_len, hidden_size}, DType::F32);

    const float* W   = weight.data_as<float>();
    float*       dst = out.data_as<float>();

    for (size_t i = 0; i < seq_len; ++i) {
        const int32_t tid = ids[i];
        if (tid < 0 || static_cast<size_t>(tid) >= vocab_size)
            throw std::out_of_range("token id " + std::to_string(tid) +
                                    " out of vocab range [0, " +
                                    std::to_string(vocab_size) + ")");
        std::memcpy(dst + i * hidden_size,
                    W   + tid * hidden_size,
                    hidden_size * sizeof(float));
    }
    return out;
}

} // namespace minillm
