// AArch64 NEON specialization of dequantize_row() for the INT8 KV-cache.
//
// This TU is compiled on AArch64 only (see CMakeLists.txt). It defines
// MINILLM_KV_Q8_HAS_NEON before pulling the header, which switches the
// portable definition in kv_cache_q8.cpp off — the linker picks up this
// override at static-library link time.
//
// Inner loop (per 16 int8 lanes):
//   int8x16_t  → int16x8_t × 2  → int32x4_t × 4  → float32x4_t × 4 × scale
//
// Tail elements (when kv_dim % 16 != 0) fall back to the scalar path.
// All production Qwen / SmolLM head dimensions are multiples of 16, so
// the tail path mostly stays dormant but is kept for correctness.
//
#if defined(__aarch64__)

#define MINILLM_KV_Q8_HAS_NEON 1

#include "minillm/kv_cache_q8.hpp"

#include <arm_neon.h>
#include <cstddef>
#include <cstdint>

namespace minillm {

void dequantize_row(float scale, std::span<const std::int8_t> data, float* out) {
    const std::size_t n = data.size();
    const float32x4_t vs = vdupq_n_f32(scale);

    std::size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        const int8x16_t q = vld1q_s8(data.data() + i);

        const int16x8_t q_lo = vmovl_s8(vget_low_s8(q));
        const int16x8_t q_hi = vmovl_s8(vget_high_s8(q));

        const int32x4_t q0 = vmovl_s16(vget_low_s16(q_lo));
        const int32x4_t q1 = vmovl_s16(vget_high_s16(q_lo));
        const int32x4_t q2 = vmovl_s16(vget_low_s16(q_hi));
        const int32x4_t q3 = vmovl_s16(vget_high_s16(q_hi));

        const float32x4_t f0 = vmulq_f32(vcvtq_f32_s32(q0), vs);
        const float32x4_t f1 = vmulq_f32(vcvtq_f32_s32(q1), vs);
        const float32x4_t f2 = vmulq_f32(vcvtq_f32_s32(q2), vs);
        const float32x4_t f3 = vmulq_f32(vcvtq_f32_s32(q3), vs);

        vst1q_f32(out + i +  0, f0);
        vst1q_f32(out + i +  4, f1);
        vst1q_f32(out + i +  8, f2);
        vst1q_f32(out + i + 12, f3);
    }
    // Scalar tail.
    for (; i < n; ++i) {
        out[i] = static_cast<float>(data[i]) * scale;
    }
}

} // namespace minillm

#endif // __aarch64__
