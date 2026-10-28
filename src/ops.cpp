#include "minillm/ops.hpp"
#include <cassert>
#include <cmath>
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

// ── RMSNorm ───────────────────────────────────────────────────────────────────

Tensor rms_norm(const Tensor& x, const Tensor& weight, float eps) {
    assert(x.dtype() == DType::F32 && weight.dtype() == DType::F32);

    const size_t hidden = x.shape().back();
    assert(weight.numel() == hidden);

    // Works for both [hidden] and [seq_len, hidden].
    const size_t batch = x.numel() / hidden;

    Tensor out(x.shape(), DType::F32);

    const float* xp = x.data_as<float>();
    const float* wp = weight.data_as<float>();
    float*       yp = out.data_as<float>();

    for (size_t b = 0; b < batch; ++b) {
        const float* row = xp + b * hidden;
        float*       dst = yp + b * hidden;

        // Mean of squares
        float ms = 0.f;
        for (size_t j = 0; j < hidden; ++j)
            ms += row[j] * row[j];
        ms /= static_cast<float>(hidden);

        const float scale = 1.f / std::sqrt(ms + eps);

        for (size_t j = 0; j < hidden; ++j)
            dst[j] = row[j] * scale * wp[j];
    }
    return out;
}

// ── Linear ────────────────────────────────────────────────────────────────────

Tensor linear(const Tensor& x, const Tensor& W, const Tensor* b) {
    assert(x.dtype() == DType::F32 && W.dtype() == DType::F32);
    assert(x.ndim() == 2 && W.ndim() == 2);

    const size_t seq_len  = x.shape()[0];
    const size_t in_feat  = x.shape()[1];
    const size_t out_feat = W.shape()[0];
    assert(W.shape()[1] == in_feat && "x columns must match W columns");

    Tensor out({seq_len, out_feat}, DType::F32);

    const float* xp = x.data_as<float>();
    const float* Wp = W.data_as<float>();
    float*       yp = out.data_as<float>();

    // y[i,j] = dot(x[i,:], W[j,:])
    for (size_t i = 0; i < seq_len; ++i) {
        const float* xrow = xp + i * in_feat;
        for (size_t j = 0; j < out_feat; ++j) {
            const float* Wrow = Wp + j * in_feat;
            float sum = 0.f;
            for (size_t k = 0; k < in_feat; ++k)
                sum += xrow[k] * Wrow[k];
            yp[i * out_feat + j] = sum;
        }
    }

    if (b) {
        assert(b->numel() == out_feat && "bias size must match out_features");
        const float* bp = b->data_as<float>();
        for (size_t i = 0; i < seq_len; ++i)
            for (size_t j = 0; j < out_feat; ++j)
                yp[i * out_feat + j] += bp[j];
    }

    return out;
}

// ── RoPE ──────────────────────────────────────────────────────────────────────

RopeCache build_rope_cache(int max_pos, int head_dim, float rope_theta) {
    assert(head_dim % 2 == 0 && "head_dim must be even for RoPE");

    const int half = head_dim / 2;

    Tensor cos_t({static_cast<size_t>(max_pos), static_cast<size_t>(half)}, DType::F32);
    Tensor sin_t({static_cast<size_t>(max_pos), static_cast<size_t>(half)}, DType::F32);

    float* cos_p = cos_t.data_as<float>();
    float* sin_p = sin_t.data_as<float>();

    for (int m = 0; m < max_pos; ++m) {
        for (int i = 0; i < half; ++i) {
            // θ_i = 1 / (rope_theta ^ (2i / head_dim))
            const float theta = 1.f / std::pow(rope_theta,
                                               2.f * static_cast<float>(i) /
                                               static_cast<float>(head_dim));
            const float angle = static_cast<float>(m) * theta;
            cos_p[m * half + i] = std::cos(angle);
            sin_p[m * half + i] = std::sin(angle);
        }
    }

    return {std::move(cos_t), std::move(sin_t)};
}

} // namespace minillm
