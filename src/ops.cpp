#include "minillm/ops.hpp"
#include <cassert>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

#ifdef __ARM_NEON
#  include <arm_neon.h>
#endif

namespace minillm {

// ── SIMD helpers ──────────────────────────────────────────────────────────────

// Dot product of two float arrays of length n.
// Uses 4-wide NEON FMA on AArch64, falls back to scalar elsewhere.
static float dot_f32(const float* __restrict__ a,
                     const float* __restrict__ b,
                     size_t n) {
#ifdef __ARM_NEON
    float32x4_t acc0 = vdupq_n_f32(0.f);
    float32x4_t acc1 = vdupq_n_f32(0.f);
    float32x4_t acc2 = vdupq_n_f32(0.f);
    float32x4_t acc3 = vdupq_n_f32(0.f);

    size_t k = 0;
    for (; k + 16 <= n; k += 16) {
        acc0 = vfmaq_f32(acc0, vld1q_f32(a + k),      vld1q_f32(b + k));
        acc1 = vfmaq_f32(acc1, vld1q_f32(a + k +  4), vld1q_f32(b + k +  4));
        acc2 = vfmaq_f32(acc2, vld1q_f32(a + k +  8), vld1q_f32(b + k +  8));
        acc3 = vfmaq_f32(acc3, vld1q_f32(a + k + 12), vld1q_f32(b + k + 12));
    }
    for (; k + 4 <= n; k += 4)
        acc0 = vfmaq_f32(acc0, vld1q_f32(a + k), vld1q_f32(b + k));

    float32x4_t acc = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
    float sum = vaddvq_f32(acc);

    for (; k < n; ++k) sum += a[k] * b[k];
    return sum;
#else
    float sum = 0.f;
    for (size_t k = 0; k < n; ++k) sum += a[k] * b[k];
    return sum;
#endif
}

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

        // Mean of squares (vectorized)
        float ms = dot_f32(row, row, hidden) / static_cast<float>(hidden);
        const float scale = 1.f / std::sqrt(ms + eps);

        // Scale + weight multiply (vectorized with NEON)
        size_t j = 0;
#ifdef __ARM_NEON
        const float32x4_t vscale = vdupq_n_f32(scale);
        for (; j + 4 <= hidden; j += 4) {
            float32x4_t r = vld1q_f32(row + j);
            float32x4_t w = vld1q_f32(wp  + j);
            vst1q_f32(dst + j, vmulq_f32(vmulq_f32(r, vscale), w));
        }
#endif
        for (; j < hidden; ++j)
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
        for (size_t j = 0; j < out_feat; ++j)
            yp[i * out_feat + j] = dot_f32(xrow, Wp + j * in_feat, in_feat);
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

// ── RoPE apply ────────────────────────────────────────────────────────────────

void rope_apply(Tensor& x, const RopeCache& cache, int pos_offset) {
    assert(x.ndim() == 2 && x.dtype() == DType::F32);

    const size_t seq_len  = x.shape()[0];
    const size_t total    = x.shape()[1];
    const size_t half     = cache.cos.shape()[1];   // head_dim / 2
    const size_t head_dim = half * 2;

    assert(total % head_dim == 0 && "x width must be a multiple of head_dim");
    const size_t n_heads = total / head_dim;

    assert(static_cast<size_t>(pos_offset) + seq_len <= cache.cos.shape()[0]
           && "positions exceed precomputed cache size");

    float*       xp    = x.data_as<float>();
    const float* cos_p = cache.cos.data_as<float>();
    const float* sin_p = cache.sin.data_as<float>();

    for (size_t s = 0; s < seq_len; ++s) {
        const size_t pos = static_cast<size_t>(pos_offset) + s;
        const float* cv  = cos_p + pos * half;
        const float* sv  = sin_p + pos * half;

        for (size_t h = 0; h < n_heads; ++h) {
            float* row = xp + s * total + h * head_dim;

            for (size_t i = 0; i < half; ++i) {
                const float x0 = row[2 * i];
                const float x1 = row[2 * i + 1];
                row[2 * i]     = x0 * cv[i] - x1 * sv[i];
                row[2 * i + 1] = x1 * cv[i] + x0 * sv[i];
            }
        }
    }
}

// ── Attention ─────────────────────────────────────────────────────────────────

void softmax_inplace(float* row, size_t n) {
    // Find max for numerical stability (prevents exp overflow).
    float max_val = row[0];
    for (size_t i = 1; i < n; ++i)
        if (row[i] > max_val) max_val = row[i];

    float sum = 0.f;
    for (size_t i = 0; i < n; ++i) {
        row[i] = std::exp(row[i] - max_val);
        sum += row[i];
    }
    const float inv_sum = 1.f / sum;
    for (size_t i = 0; i < n; ++i)
        row[i] *= inv_sum;
}

Tensor scaled_dot_product_attention(const Tensor& q,
                                     const Tensor& k,
                                     const Tensor& v,
                                     bool causal,
                                     int  q_offset) {
    assert(q.ndim() == 2 && k.ndim() == 2 && v.ndim() == 2);
    assert(q.dtype() == DType::F32 && k.dtype() == DType::F32 && v.dtype() == DType::F32);

    const size_t q_len   = q.shape()[0];
    const size_t kv_len  = k.shape()[0];
    const size_t head_dim = q.shape()[1];
    const float  scale   = 1.f / std::sqrt(static_cast<float>(head_dim));

    assert(k.shape()[1] == head_dim);
    assert(v.shape()[0] == kv_len && v.shape()[1] == head_dim);

    const float* qp = q.data_as<float>();
    const float* kp = k.data_as<float>();
    const float* vp = v.data_as<float>();

    // scores[i,j] = dot(q[i], k[j]) * scale
    Tensor scores({q_len, kv_len}, DType::F32);
    float* sp = scores.data_as<float>();

    for (size_t i = 0; i < q_len; ++i) {
        for (size_t j = 0; j < kv_len; ++j) {
            float dot = 0.f;
            for (size_t d = 0; d < head_dim; ++d)
                dot += qp[i * head_dim + d] * kp[j * head_dim + d];
            sp[i * kv_len + j] = dot * scale;
        }
    }

    // Causal mask: q[i] is at position q_offset+i; mask j > q_offset+i
    if (causal) {
        for (size_t i = 0; i < q_len; ++i)
            for (size_t j = static_cast<size_t>(q_offset) + i + 1; j < kv_len; ++j)
                sp[i * kv_len + j] = -std::numeric_limits<float>::infinity();
    }

    for (size_t i = 0; i < q_len; ++i)
        softmax_inplace(sp + i * kv_len, kv_len);

    Tensor out({q_len, head_dim}, DType::F32);
    float* op = out.data_as<float>();

    for (size_t i = 0; i < q_len; ++i) {
        for (size_t d = 0; d < head_dim; ++d) {
            float acc = 0.f;
            for (size_t j = 0; j < kv_len; ++j)
                acc += sp[i * kv_len + j] * vp[j * head_dim + d];
            op[i * head_dim + d] = acc;
        }
    }

    return out;
}

// ── Elementwise ops ───────────────────────────────────────────────────────────

void add_inplace(Tensor& a, const Tensor& b) {
    assert(a.dtype() == DType::F32 && b.dtype() == DType::F32);
    assert(a.shape() == b.shape());
    float*       ap = a.data_as<float>();
    const float* bp = b.data_as<float>();
    const size_t n  = a.numel();
    for (size_t i = 0; i < n; ++i) ap[i] += bp[i];
}

// ── MLP ───────────────────────────────────────────────────────────────────────

Tensor swiglu_mlp(const Tensor& x,
                   const Tensor& W_gate,
                   const Tensor& W_up,
                   const Tensor& W_down) {
    assert(x.dtype() == DType::F32);

    // Parallel gate and up projections
    Tensor gate = linear(x, W_gate);   // [seq, intermediate]
    Tensor up   = linear(x, W_up);     // [seq, intermediate]

    // SwiGLU: gate[i] = silu(gate[i]) * up[i]  (in-place on gate)
    float*       gp  = gate.data_as<float>();
    const float* up_p = up.data_as<float>();
    const size_t n   = gate.numel();

    for (size_t i = 0; i < n; ++i)
        gp[i] = silu(gp[i]) * up_p[i];

    // Down projection
    return linear(gate, W_down);       // [seq, hidden]
}

// ── GQA ───────────────────────────────────────────────────────────────────────

// Copy columns [col_start, col_start+width) from every row of x.
// Returns a new owning [n_rows, width] F32 tensor.
static Tensor col_slice(const Tensor& x, size_t col_start, size_t width) {
    assert(x.ndim() == 2 && x.dtype() == DType::F32);
    const size_t n_rows = x.shape()[0];
    const size_t n_cols = x.shape()[1];
    assert(col_start + width <= n_cols);

    Tensor out({n_rows, width}, DType::F32);
    const float* src = x.data_as<float>();
    float*       dst = out.data_as<float>();
    for (size_t r = 0; r < n_rows; ++r)
        std::memcpy(dst + r * width,
                    src + r * n_cols + col_start,
                    width * sizeof(float));
    return out;
}

Tensor gqa_attention(const Tensor& x,
                     const Tensor& Wq,
                     const Tensor& Wk,
                     const Tensor& Wv,
                     const Tensor& Wo,
                     const RopeCache& rope,
                     int n_heads,
                     int n_kv_heads,
                     int pos_offset) {
    assert(x.ndim() == 2 && x.dtype() == DType::F32);
    assert(n_heads % n_kv_heads == 0);

    const size_t seq_len   = x.shape()[0];
    const size_t head_dim  = Wq.shape()[0] / static_cast<size_t>(n_heads);
    const int    kv_groups = n_heads / n_kv_heads;

    // 1. Linear projections
    Tensor Q = linear(x, Wq);   // [seq, n_heads    * head_dim]
    Tensor K = linear(x, Wk);   // [seq, n_kv_heads * head_dim]
    Tensor V = linear(x, Wv);   // [seq, n_kv_heads * head_dim]

    // 2. Apply RoPE in-place
    rope_apply(Q, rope, pos_offset);
    rope_apply(K, rope, pos_offset);

    // 3. Per-head attention, results written into attn_out
    Tensor attn_out({seq_len, static_cast<size_t>(n_heads) * head_dim}, DType::F32);
    float* out_p = attn_out.data_as<float>();

    for (int h = 0; h < n_heads; ++h) {
        const size_t kv_h = static_cast<size_t>(h / kv_groups);

        Tensor q_h  = col_slice(Q, static_cast<size_t>(h)  * head_dim, head_dim);
        Tensor k_h  = col_slice(K, kv_h * head_dim, head_dim);
        Tensor v_h  = col_slice(V, kv_h * head_dim, head_dim);

        Tensor out_h = scaled_dot_product_attention(q_h, k_h, v_h, /*causal=*/true);

        // Write head output into the right column slice of attn_out
        const float* src = out_h.data_as<float>();
        for (size_t s = 0; s < seq_len; ++s)
            std::memcpy(out_p + s * static_cast<size_t>(n_heads) * head_dim
                               + static_cast<size_t>(h) * head_dim,
                        src + s * head_dim,
                        head_dim * sizeof(float));
    }

    // 4. Output projection
    return linear(attn_out, Wo);
}

// ── Quantization ──────────────────────────────────────────────────────────────

QuantTensor quantize(const Tensor& w, DType qdtype) {
    assert(w.ndim() == 2 && w.dtype() == DType::F32);
    assert(qdtype == DType::I8 || qdtype == DType::I4);

    const size_t out_feat = w.shape()[0];
    const size_t in_feat  = w.shape()[1];
    const float* wp       = w.data_as<float>();
    const float  qmax     = (qdtype == DType::I8) ? 127.f : 7.f;

    std::vector<float> scales(out_feat);
    for (size_t j = 0; j < out_feat; ++j) {
        float abs_max = 0.f;
        const float* row = wp + j * in_feat;
        for (size_t k = 0; k < in_feat; ++k)
            if (std::abs(row[k]) > abs_max) abs_max = std::abs(row[k]);
        scales[j] = (abs_max > 0.f) ? abs_max / qmax : 1.f;
    }

    Tensor data(w.shape(), qdtype);

    if (qdtype == DType::I8) {
        int8_t* qp = reinterpret_cast<int8_t*>(data.data_as<uint8_t>());
        for (size_t j = 0; j < out_feat; ++j) {
            const float inv = 1.f / scales[j];
            const float* row = wp + j * in_feat;
            int8_t* qrow     = qp + j * in_feat;
            for (size_t k = 0; k < in_feat; ++k) {
                float v = std::round(row[k] * inv);
                v = std::max(-127.f, std::min(127.f, v));
                qrow[k] = static_cast<int8_t>(v);
            }
        }
    } else {  // I4 — nibble-packed, 2 values per byte, high nibble first
        uint8_t* qp = data.data_as<uint8_t>();
        const size_t bytes_per_row = (in_feat + 1) / 2;
        auto clamp4 = [](float v) -> int8_t {
            int iv = static_cast<int>(std::round(v));
            return static_cast<int8_t>(std::max(-7, std::min(7, iv)));
        };
        for (size_t j = 0; j < out_feat; ++j) {
            const float inv = 1.f / scales[j];
            const float* row = wp + j * in_feat;
            uint8_t* qrow    = qp + j * bytes_per_row;
            for (size_t k = 0; k < in_feat; k += 2) {
                int8_t hi = clamp4(row[k] * inv);
                int8_t lo = (k + 1 < in_feat) ? clamp4(row[k + 1] * inv) : 0;
                qrow[k / 2] = static_cast<uint8_t>(
                    (static_cast<uint8_t>(hi) & 0x0Fu) << 4u |
                    (static_cast<uint8_t>(lo) & 0x0Fu));
            }
        }
    }

    return {std::move(data), std::move(scales)};
}

Tensor linear_q(const Tensor& x, const QuantTensor& Wq) {
    assert(x.ndim() == 2 && x.dtype() == DType::F32);
    assert(Wq.data.ndim() == 2);

    const size_t seq_len  = x.shape()[0];
    const size_t in_feat  = x.shape()[1];
    const size_t out_feat = Wq.data.shape()[0];
    assert(Wq.scales.size() == out_feat);

    Tensor out({seq_len, out_feat}, DType::F32);
    const float* xp = x.data_as<float>();
    float*       yp = out.data_as<float>();

    // Dequantize one output row at a time to avoid allocating a full F32 copy.
    std::vector<float> row(in_feat);

    if (Wq.data.dtype() == DType::I8) {
        const auto* qp = reinterpret_cast<const int8_t*>(Wq.data.data_as<uint8_t>());
        for (size_t j = 0; j < out_feat; ++j) {
            const float   s    = Wq.scales[j];
            const int8_t* qrow = qp + j * in_feat;
            for (size_t k = 0; k < in_feat; ++k)
                row[k] = static_cast<float>(qrow[k]) * s;
            for (size_t i = 0; i < seq_len; ++i)
                yp[i * out_feat + j] = dot_f32(xp + i * in_feat, row.data(), in_feat);
        }
    } else {  // I4
        const size_t   bytes_per_row = (in_feat + 1) / 2;
        const uint8_t* qp            = Wq.data.data_as<uint8_t>();
        for (size_t j = 0; j < out_feat; ++j) {
            const float    s    = Wq.scales[j];
            const uint8_t* qrow = qp + j * bytes_per_row;
            for (size_t k = 0; k < in_feat; k += 2) {
                uint8_t byte = qrow[k / 2];
                row[k] = static_cast<float>(static_cast<int8_t>(byte & 0xF0u) >> 4) * s;
                if (k + 1 < in_feat)
                    row[k + 1] = static_cast<float>(
                        static_cast<int8_t>(static_cast<uint8_t>(byte << 4u)) >> 4) * s;
            }
            for (size_t i = 0; i < seq_len; ++i)
                yp[i * out_feat + j] = dot_f32(xp + i * in_feat, row.data(), in_feat);
        }
    }

    return out;
}

Tensor dequantize(const QuantTensor& Wq) {
    const size_t out_feat = Wq.data.shape()[0];
    const size_t in_feat  = Wq.data.shape()[1];
    assert(Wq.scales.size() == out_feat);
    Tensor out({out_feat, in_feat}, DType::F32);
    float* yp = out.data_as<float>();

    if (Wq.data.dtype() == DType::I8) {
        const auto* qp = reinterpret_cast<const int8_t*>(Wq.data.data_as<uint8_t>());
        for (size_t j = 0; j < out_feat; ++j) {
            const float s = Wq.scales[j];
            for (size_t k = 0; k < in_feat; ++k)
                yp[j * in_feat + k] = static_cast<float>(qp[j * in_feat + k]) * s;
        }
    } else {
        const size_t   bytes_per_row = (in_feat + 1) / 2;
        const uint8_t* qp            = Wq.data.data_as<uint8_t>();
        for (size_t j = 0; j < out_feat; ++j) {
            const float    s    = Wq.scales[j];
            const uint8_t* qrow = qp + j * bytes_per_row;
            for (size_t k = 0; k < in_feat; k += 2) {
                uint8_t byte = qrow[k / 2];
                yp[j * in_feat + k] =
                    static_cast<float>(static_cast<int8_t>(byte & 0xF0u) >> 4) * s;
                if (k + 1 < in_feat)
                    yp[j * in_feat + k + 1] =
                        static_cast<float>(static_cast<int8_t>(static_cast<uint8_t>(byte << 4u)) >> 4)
                        * s;
            }
        }
    }
    return out;
}

} // namespace minillm
