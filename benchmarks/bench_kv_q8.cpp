// Benchmark for the T26 INT8 KV-cache utility.
//
// Three numbers per cell:
//   1. F32  append  — baseline cost of pushing one token row into the
//      portable KVCache (just an insert into two std::vectors).
//   2. Q8   append  — same but through KVCacheQ8::append(), which runs
//      per-row scale discovery + symmetric int8 quantization.
//   3. Dequant throughput — ms to materialise the full cache back to
//      F32 via KVCacheQ8::dequantize_all_k(). This is the number that
//      matters for a decode step that consumes the whole K history.
//
// Shapes mirror SmolLM2-135M (kv_dim = n_kv_heads * head_dim = 3 * 64 = 192)
// and Qwen2.5-0.5B (kv_dim = 2 * 64 = 128).
//
// Reproduce (Release build, Apple Silicon):
//   cmake --build build -j
//   ./build/benchmarks/bench_kv_q8
//
// Output also prints bytes stored vs bytes-if-F32, so the 4× memory
// win from INT8 is visible alongside the timing.
//
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "minillm/kv_cache_q8.hpp"

using namespace minillm;
using clk = std::chrono::steady_clock;

static double now_ms() {
    return std::chrono::duration<double, std::milli>(clk::now().time_since_epoch()).count();
}

static void rand_fill(std::vector<float>& v) {
    for (auto& x : v) {
        x = (static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX)) - 0.5f;
    }
}

struct Row {
    std::string label;
    double      f32_append_ms;
    double      q8_append_ms;
    double      q8_dequant_full_ms;
    std::size_t bytes_f32;
    std::size_t bytes_q8;
};

static Row run_cell(std::size_t kv_dim, std::size_t n_tokens, const std::string& label) {
    // Shared input: one random row per token, reused across both impls.
    std::vector<std::vector<float>> rows(n_tokens, std::vector<float>(kv_dim));
    for (auto& r : rows) rand_fill(r);

    // ── F32 baseline: push into two std::vectors. ────────────────────────────
    std::vector<float> f32_k, f32_v;
    f32_k.reserve(n_tokens * kv_dim);
    f32_v.reserve(n_tokens * kv_dim);
    const double t_f32_0 = now_ms();
    for (std::size_t t = 0; t < n_tokens; ++t) {
        f32_k.insert(f32_k.end(), rows[t].begin(), rows[t].end());
        f32_v.insert(f32_v.end(), rows[t].begin(), rows[t].end());
    }
    const double f32_total_ms = now_ms() - t_f32_0;

    // ── Q8 append: scale discovery + quantize per token. ─────────────────────
    KVCacheQ8::Layer layer;
    layer.kv_dim = kv_dim;
    layer.k_data.reserve(n_tokens * kv_dim);
    layer.v_data.reserve(n_tokens * kv_dim);
    layer.k_scales.reserve(n_tokens);
    layer.v_scales.reserve(n_tokens);
    const double t_q8_0 = now_ms();
    for (std::size_t t = 0; t < n_tokens; ++t) {
        layer.append(rows[t].data(), rows[t].data());
    }
    const double q8_total_ms = now_ms() - t_q8_0;

    // ── Q8 dequantize full cache (NEON path on AArch64). ─────────────────────
    std::vector<float> q8_back(n_tokens * kv_dim);
    // One warm-up iteration.
    layer.dequantize_all_k(q8_back.data());
    const int deq_reps = 32;
    const double t_deq_0 = now_ms();
    for (int r = 0; r < deq_reps; ++r) {
        layer.dequantize_all_k(q8_back.data());
    }
    const double q8_deq_ms = (now_ms() - t_deq_0) / deq_reps;

    const std::size_t bytes_f32 = 2 * n_tokens * kv_dim * sizeof(float);
    return Row{
        label,
        f32_total_ms / static_cast<double>(n_tokens),
        q8_total_ms  / static_cast<double>(n_tokens),
        q8_deq_ms,
        bytes_f32,
        layer.bytes(), // K int8 + V int8 + K scales + V scales
    };
}

int main() {
    std::srand(42);

    std::printf("%-30s | %10s | %10s | %12s | %12s | %10s\n",
                "shape", "F32 push", "Q8 append", "Q8 dequant", "bytes F32", "bytes Q8");
    std::printf("%-30s | %10s | %10s | %12s | %12s | %10s\n",
                "", "(ms/tok)", "(ms/tok)", "(ms full)", "", "");
    std::printf("%s\n", std::string(96, '-').c_str());

    struct Cell {
        std::size_t kv_dim;
        std::size_t n_tokens;
        const char* label;
    };
    const Cell cells[] = {
        {128,  256,  "qwen2.5-0.5b  kv=128 n=256"},
        {128, 1024,  "qwen2.5-0.5b  kv=128 n=1024"},
        {128, 4096,  "qwen2.5-0.5b  kv=128 n=4096"},
        {192,  256,  "smollm2-135m  kv=192 n=256"},
        {192, 1024,  "smollm2-135m  kv=192 n=1024"},
        {192, 4096,  "smollm2-135m  kv=192 n=4096"},
    };

    for (const auto& c : cells) {
        const Row r = run_cell(c.kv_dim, c.n_tokens, c.label);
        std::printf("%-30s | %10.4f | %10.4f | %12.4f | %12zu | %10zu\n",
                    r.label.c_str(),
                    r.f32_append_ms,
                    r.q8_append_ms,
                    r.q8_dequant_full_ms,
                    r.bytes_f32,
                    r.bytes_q8);
    }

    std::printf("\nMemory ratio target: Q8 ≈ F32 / 4 (plus one fp32 scale per token).\n");
    return 0;
}
