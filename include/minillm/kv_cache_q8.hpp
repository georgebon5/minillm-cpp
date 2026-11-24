#pragma once
//
// INT8 KV-cache utility (T26, cross-contribution with the GreekKV study).
//
// Symmetric per-token quantization along the sequence axis:
//   row k_data[t, :] ≈ k_scales[t] * int8(k_data[t, :])
//
// The scale is max(|row|) / 127. Each new token appended to the cache gets
// its own scale, so the per-token quantization error is bounded by half of
// a single step of that row's own dynamic range — which is the standard
// KIVI-K / KIVI-V layout.
//
// This header is intentionally decoupled from the live generation path
// (model.hpp still uses the F32 KVCache). It exposes:
//   - make_kvcache_q8(cfg) : allocate an empty per-layer structure.
//   - KVCacheQ8::Layer::append(k_row, v_row) : push a new F32 token row
//     through quantization and store (int8, scale).
//   - KVCacheQ8::Layer::dequantize_all_k / _v : materialise back to F32,
//     with a NEON path on AArch64 and a scalar fallback everywhere else.
//
// Numerical contract (asserted by test_kv_cache_q8):
//   max_abs(row - dequant(row))  ≤  k_scales[row] + 1e-6 f
// which is the usual symmetric-quant round-trip bound.
//
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "minillm/model_config.hpp"

namespace minillm {

struct KVCacheQ8 {
    struct Layer {
        std::vector<int8_t> k_data;    // row-major [seq_len * kv_dim], int8
        std::vector<int8_t> v_data;    // row-major [seq_len * kv_dim], int8
        std::vector<float>  k_scales;  // one scale per row (per-token)
        std::vector<float>  v_scales;
        std::size_t         kv_dim  = 0;
        std::size_t         seq_len = 0;

        // Quantize one F32 token row (length kv_dim) and append.
        // Costs O(kv_dim) for scale discovery + O(kv_dim) for quantization.
        void append(const float* k_row, const float* v_row);

        // Dequantize the entire K cache back to a contiguous
        // [seq_len * kv_dim] F32 buffer. Uses NEON on AArch64 when the
        // row stride is a multiple of 16, scalar elsewhere.
        void dequantize_all_k(float* out) const;
        void dequantize_all_v(float* out) const;

        // Byte footprint (int8 bodies + fp32 scales). Handy for the
        // benchmark output and the GreekKV capacity model.
        std::size_t bytes() const {
            return k_data.size() * sizeof(int8_t)
                 + v_data.size() * sizeof(int8_t)
                 + (k_scales.size() + v_scales.size()) * sizeof(float);
        }
    };

    std::vector<Layer> layers;
};

// Allocate an empty INT8 KV cache sized for cfg (zero tokens stored).
KVCacheQ8 make_kvcache_q8(const ModelConfig& cfg);

// Scalar reference used by the fallback path *and* by tests to assert
// parity against the NEON path. Exposed in the header so tests can call
// it without reaching into the .cpp.
//
//   scale  : the symmetric per-row scale produced by append().
//   data   : int8 row of length kv_dim.
//   out    : F32 row of length kv_dim.
void dequantize_row_scalar(float scale, std::span<const int8_t> data, float* out);

// Same contract as dequantize_row_scalar but takes the NEON path on
// AArch64 when kv_dim % 16 == 0. Falls back to the scalar path otherwise.
void dequantize_row(float scale, std::span<const int8_t> data, float* out);

} // namespace minillm
