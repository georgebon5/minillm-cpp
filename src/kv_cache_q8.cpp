// Scalar implementation of the INT8 KV-cache utility declared in
// include/minillm/kv_cache_q8.hpp. The NEON specialization of
// dequantize_row() is added in a separate translation-unit-local
// override (T26.2) — this file ships the portable path and the public
// entry points.
//
#include "minillm/kv_cache_q8.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace minillm {

namespace {

// Per-token symmetric scale: s = max(|row|) / 127, clamped to a floor
// so an all-zero row does not divide by zero on append.
float row_scale_i8(const float* row, std::size_t n) {
    float absmax = 0.0f;
    for (std::size_t i = 0; i < n; ++i) {
        const float a = std::fabs(row[i]);
        if (a > absmax) absmax = a;
    }
    const float s = absmax / 127.0f;
    return s < 1e-9f ? 1e-9f : s;
}

void quantize_row_i8(const float* row, std::size_t n, float scale, std::int8_t* out) {
    const float inv = 1.0f / scale;
    for (std::size_t i = 0; i < n; ++i) {
        const float q = std::round(row[i] * inv);
        const float c = std::clamp(q, -127.0f, 127.0f);
        out[i] = static_cast<std::int8_t>(c);
    }
}

} // namespace

void dequantize_row_scalar(float scale, std::span<const std::int8_t> data, float* out) {
    for (std::size_t i = 0; i < data.size(); ++i) {
        out[i] = static_cast<float>(data[i]) * scale;
    }
}

// Default (portable) entry point. On AArch64 a NEON override replaces
// this symbol via a separate translation unit (see T26.2 commit).
#if !defined(MINILLM_KV_Q8_HAS_NEON)
void dequantize_row(float scale, std::span<const std::int8_t> data, float* out) {
    dequantize_row_scalar(scale, data, out);
}
#endif

void KVCacheQ8::Layer::append(const float* k_row, const float* v_row) {
    const std::size_t n = kv_dim;
    const std::size_t off = k_data.size();
    k_data.resize(off + n);
    v_data.resize(off + n);

    const float sk = row_scale_i8(k_row, n);
    const float sv = row_scale_i8(v_row, n);
    quantize_row_i8(k_row, n, sk, k_data.data() + off);
    quantize_row_i8(v_row, n, sv, v_data.data() + off);
    k_scales.push_back(sk);
    v_scales.push_back(sv);
    ++seq_len;
}

void KVCacheQ8::Layer::dequantize_all_k(float* out) const {
    for (std::size_t t = 0; t < seq_len; ++t) {
        dequantize_row(
            k_scales[t],
            std::span<const std::int8_t>(k_data.data() + t * kv_dim, kv_dim),
            out + t * kv_dim);
    }
}

void KVCacheQ8::Layer::dequantize_all_v(float* out) const {
    for (std::size_t t = 0; t < seq_len; ++t) {
        dequantize_row(
            v_scales[t],
            std::span<const std::int8_t>(v_data.data() + t * kv_dim, kv_dim),
            out + t * kv_dim);
    }
}

KVCacheQ8 make_kvcache_q8(const ModelConfig& cfg) {
    KVCacheQ8 kv;
    kv.layers.resize(static_cast<std::size_t>(cfg.num_hidden_layers));
    const std::size_t kv_dim = static_cast<std::size_t>(cfg.num_key_value_heads) *
                               static_cast<std::size_t>(cfg.head_dim);
    for (auto& l : kv.layers) l.kv_dim = kv_dim;
    return kv;
}

} // namespace minillm
