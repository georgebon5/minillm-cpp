#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include "minillm/ops.hpp"
#include "minillm/tensor.hpp"

using namespace minillm;
using clk = std::chrono::steady_clock;

static double now_ms() {
    return std::chrono::duration<double, std::milli>(clk::now().time_since_epoch()).count();
}

// Fill tensor with random floats in [-0.5, 0.5]
static void rand_fill(Tensor& t) {
    float* p = t.data_as<float>();
    for (size_t i = 0; i < t.numel(); ++i)
        p[i] = (static_cast<float>(std::rand()) / RAND_MAX) - 0.5f;
}

static double bench_linear(size_t seq, size_t in_f, size_t out_f, int reps) {
    Tensor x({seq, in_f},     DType::F32);
    Tensor W({out_f, in_f},   DType::F32);
    rand_fill(x); rand_fill(W);

    // Warm-up
    for (int i = 0; i < 3; ++i) { auto r = linear(x, W); (void)r; }

    double t0 = now_ms();
    for (int i = 0; i < reps; ++i) { auto r = linear(x, W); (void)r; }
    return (now_ms() - t0) / reps;
}

static double bench_rms_norm(size_t seq, size_t hidden, int reps) {
    Tensor x({seq, hidden},    DType::F32);
    Tensor w({hidden},         DType::F32);
    rand_fill(x); rand_fill(w);

    for (int i = 0; i < 3; ++i) { auto r = rms_norm(x, w, 1e-5f); (void)r; }

    double t0 = now_ms();
    for (int i = 0; i < reps; ++i) { auto r = rms_norm(x, w, 1e-5f); (void)r; }
    return (now_ms() - t0) / reps;
}

int main() {
    std::srand(42);

    // SmolLM2-135M dimensions: hidden=576, intermediate=1536, n_heads=9, head_dim=64
    constexpr size_t H = 576, INT = 1536, V = 49152;

    std::printf("%-40s  %8s\n", "benchmark", "ms/call");
    std::printf("%s\n", std::string(52, '-').c_str());

    auto row = [](const char* name, double ms) {
        std::printf("%-40s  %8.4f\n", name, ms);
    };

    row("linear  seq=1   576→1536  (gate/up)", bench_linear(1,  H,   INT, 200));
    row("linear  seq=1   576→576   (Wq)",      bench_linear(1,  H,   H,   200));
    row("linear  seq=1   576→49152 (lm_head)", bench_linear(1,  H,   V,    50));
    row("linear  seq=32  576→1536",            bench_linear(32, H,   INT,  50));
    row("linear  seq=128 576→1536",            bench_linear(128,H,   INT,  20));
    row("rms_norm seq=1  hidden=576",           bench_rms_norm(1,  H, 500));
    row("rms_norm seq=128 hidden=576",          bench_rms_norm(128,H,  50));
}
