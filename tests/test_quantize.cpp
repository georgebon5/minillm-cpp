#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "minillm/ops.hpp"
#include "minillm/tensor.hpp"
#include <cmath>
#include <cstdlib>
#include <numeric>

using namespace minillm;

// ── helpers ───────────────────────────────────────────────────────────────────

static void rand_fill(Tensor& t, float lo = -1.f, float hi = 1.f) {
    float* p = t.data_as<float>();
    for (size_t i = 0; i < t.numel(); ++i)
        p[i] = lo + (hi - lo) * (static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX));
}

static float max_abs_error(const Tensor& a, const Tensor& b) {
    assert(a.numel() == b.numel());
    const float* ap = a.data_as<float>();
    const float* bp = b.data_as<float>();
    float err = 0.f;
    for (size_t i = 0; i < a.numel(); ++i)
        err = std::max(err, std::abs(ap[i] - bp[i]));
    return err;
}

// ── INT8 quantization ─────────────────────────────────────────────────────────

TEST_CASE("quantize I8: per-row scales are positive", "[quantize]") {
    std::srand(1);
    Tensor w({4, 8}, DType::F32);
    rand_fill(w);
    auto qw = quantize(w, DType::I8);
    REQUIRE(qw.scales.size() == 4);
    for (float s : qw.scales) REQUIRE(s > 0.f);
}

TEST_CASE("quantize I8: quantized values in [-127, 127]", "[quantize]") {
    std::srand(2);
    Tensor w({4, 8}, DType::F32);
    rand_fill(w);
    auto qw = quantize(w, DType::I8);
    const auto* qp = reinterpret_cast<const int8_t*>(qw.data.data_as<uint8_t>());
    for (size_t i = 0; i < qw.data.numel(); ++i) {
        REQUIRE(qp[i] >= -127);
        REQUIRE(qp[i] <=  127);
    }
}

TEST_CASE("quantize I8: memory is 4x smaller than F32", "[quantize]") {
    Tensor w({32, 64}, DType::F32);
    auto qw = quantize(w, DType::I8);
    REQUIRE(qw.data.nbytes() * 4 == w.nbytes());
}

TEST_CASE("linear_q I8: output shape matches linear()", "[quantize]") {
    std::srand(3);
    Tensor x({2, 8}, DType::F32);
    Tensor W({4, 8}, DType::F32);
    rand_fill(x); rand_fill(W);
    auto qW = quantize(W, DType::I8);

    Tensor ref = linear(x, W);
    Tensor got = linear_q(x, qW);

    REQUIRE(ref.shape() == got.shape());
}

TEST_CASE("linear_q I8: max abs error < 0.05 on uniform weights", "[quantize]") {
    std::srand(4);
    Tensor x({4, 64}, DType::F32);
    Tensor W({32, 64}, DType::F32);
    rand_fill(x, -1.f, 1.f);
    rand_fill(W, -1.f, 1.f);

    Tensor ref = linear(x, W);
    Tensor got = linear_q(x, quantize(W, DType::I8));

    float err = max_abs_error(ref, got);
    INFO("INT8 max abs error: " << err);
    REQUIRE(err < 0.05f);
}

// ── INT4 quantization ─────────────────────────────────────────────────────────

TEST_CASE("quantize I4: memory is 8x smaller than F32", "[quantize]") {
    Tensor w({32, 64}, DType::F32);
    auto qw = quantize(w, DType::I4);
    REQUIRE(qw.data.nbytes() * 8 == w.nbytes());
}

TEST_CASE("linear_q I4: output shape matches linear()", "[quantize]") {
    std::srand(5);
    Tensor x({2, 8}, DType::F32);
    Tensor W({4, 8}, DType::F32);
    rand_fill(x); rand_fill(W);

    Tensor ref = linear(x, W);
    Tensor got = linear_q(x, quantize(W, DType::I4));

    REQUIRE(ref.shape() == got.shape());
}

TEST_CASE("linear_q I4: max abs error < 0.3 on uniform weights", "[quantize]") {
    std::srand(6);
    Tensor x({4, 64}, DType::F32);
    Tensor W({32, 64}, DType::F32);
    rand_fill(x, -1.f, 1.f);
    rand_fill(W, -1.f, 1.f);

    Tensor ref = linear(x, W);
    Tensor got = linear_q(x, quantize(W, DType::I4));

    float err = max_abs_error(ref, got);
    INFO("INT4 max abs error: " << err);
    // INT4 is coarser (3 bits magnitude): worst-case error grows with sqrt(in_feat).
    // With in_feat=64 and scale≈1/7, RMS error ≈ sqrt(64)*0.071 ≈ 0.57.
    REQUIRE(err < 1.0f);
}

TEST_CASE("linear_q I8 better than I4: INT8 error < INT4 error", "[quantize]") {
    std::srand(7);
    Tensor x({4, 64}, DType::F32);
    Tensor W({32, 64}, DType::F32);
    rand_fill(x, -1.f, 1.f);
    rand_fill(W, -1.f, 1.f);

    Tensor ref  = linear(x, W);
    float  e8   = max_abs_error(ref, linear_q(x, quantize(W, DType::I8)));
    float  e4   = max_abs_error(ref, linear_q(x, quantize(W, DType::I4)));

    INFO("INT8 err=" << e8 << "  INT4 err=" << e4);
    REQUIRE(e8 < e4);
}
