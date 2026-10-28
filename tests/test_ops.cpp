#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "minillm/ops.hpp"
#include <array>
#include <cmath>

using namespace minillm;

// ── Embedding lookup ──────────────────────────────────────────────────────────

// Build a small embedding matrix: vocab=5, hidden=3.
// Row i = {i*10+1, i*10+2, i*10+3} for easy visual verification.
static Tensor make_embed_weight() {
    Tensor w({5, 3}, DType::F32);
    float* p = w.data_as<float>();
    for (int i = 0; i < 5; ++i) {
        p[i * 3 + 0] = float(i * 10 + 1);
        p[i * 3 + 1] = float(i * 10 + 2);
        p[i * 3 + 2] = float(i * 10 + 3);
    }
    return w;
}

TEST_CASE("embedding_lookup: output shape", "[ops][embedding]") {
    auto w = make_embed_weight();
    std::array<int32_t, 4> ids = {0, 2, 1, 4};
    Tensor out = embedding_lookup(w, ids);
    REQUIRE(out.shape() == std::vector<size_t>{4, 3});
    REQUIRE(out.dtype() == DType::F32);
}

TEST_CASE("embedding_lookup: values match embedding rows", "[ops][embedding]") {
    auto w = make_embed_weight();
    std::array<int32_t, 3> ids = {0, 3, 4};
    Tensor out = embedding_lookup(w, ids);

    // Row 0: token 0 → {1, 2, 3}
    REQUIRE(out.at<float>({0, 0}) == 1.f);
    REQUIRE(out.at<float>({0, 1}) == 2.f);
    REQUIRE(out.at<float>({0, 2}) == 3.f);

    // Row 1: token 3 → {31, 32, 33}
    REQUIRE(out.at<float>({1, 0}) == 31.f);

    // Row 2: token 4 → {41, 42, 43}
    REQUIRE(out.at<float>({2, 2}) == 43.f);
}

TEST_CASE("embedding_lookup: repeated token", "[ops][embedding]") {
    auto w = make_embed_weight();
    std::array<int32_t, 3> ids = {2, 2, 2};
    Tensor out = embedding_lookup(w, ids);
    // All three rows should be identical copies of row 2: {21, 22, 23}
    for (int r = 0; r < 3; ++r)
        REQUIRE(out.at<float>({(size_t)r, 0}) == 21.f);
}

TEST_CASE("embedding_lookup: out-of-range id throws", "[ops][embedding]") {
    auto w = make_embed_weight();
    std::array<int32_t, 1> ids = {99};
    REQUIRE_THROWS_AS(embedding_lookup(w, ids), std::out_of_range);
}

TEST_CASE("embedding_lookup: negative id throws", "[ops][embedding]") {
    auto w = make_embed_weight();
    std::array<int32_t, 1> ids = {-1};
    REQUIRE_THROWS_AS(embedding_lookup(w, ids), std::out_of_range);
}

// ── RMSNorm ───────────────────────────────────────────────────────────────────

TEST_CASE("rms_norm: output shape matches input", "[ops][norm]") {
    Tensor x({3, 4}, DType::F32);
    Tensor w({4},    DType::F32);
    for (size_t i = 0; i < 4; ++i) w.data_as<float>()[i] = 1.f;
    Tensor y = rms_norm(x, w, 1e-5f);
    REQUIRE(y.shape() == x.shape());
    REQUIRE(y.dtype() == DType::F32);
}

TEST_CASE("rms_norm: 1-D vector with unit weights", "[ops][norm]") {
    // x = [3, 4], w = [1, 1], eps = 0
    // RMS = sqrt((9+16)/2) = sqrt(12.5)
    // y = x / RMS
    Tensor x({2}, DType::F32);
    x.data_as<float>()[0] = 3.f;
    x.data_as<float>()[1] = 4.f;

    Tensor w({2}, DType::F32);
    w.data_as<float>()[0] = 1.f;
    w.data_as<float>()[1] = 1.f;

    Tensor y = rms_norm(x, w, 0.f);

    float rms = std::sqrt(12.5f);
    REQUIRE(y.at<float>({0}) == Catch::Approx(3.f / rms).epsilon(1e-5));
    REQUIRE(y.at<float>({1}) == Catch::Approx(4.f / rms).epsilon(1e-5));
}

TEST_CASE("rms_norm: scale weights are applied", "[ops][norm]") {
    // x = [1, 1], w = [2, 3], eps = 0
    // RMS = sqrt((1+1)/2) = 1.0
    // y = [1*2, 1*3] = [2, 3]
    Tensor x({2}, DType::F32);
    x.data_as<float>()[0] = 1.f;
    x.data_as<float>()[1] = 1.f;

    Tensor w({2}, DType::F32);
    w.data_as<float>()[0] = 2.f;
    w.data_as<float>()[1] = 3.f;

    Tensor y = rms_norm(x, w, 0.f);
    REQUIRE(y.at<float>({0}) == Catch::Approx(2.f).epsilon(1e-5));
    REQUIRE(y.at<float>({1}) == Catch::Approx(3.f).epsilon(1e-5));
}

TEST_CASE("rms_norm: each row is normalized independently", "[ops][norm]") {
    // Two rows: [3,4] and [6,8]. Both should normalize to same direction.
    // [6,8] = 2*[3,4] so after norm both rows should be identical.
    Tensor x({2, 2}, DType::F32);
    float* p = x.data_as<float>();
    p[0] = 3.f; p[1] = 4.f;
    p[2] = 6.f; p[3] = 8.f;

    Tensor w({2}, DType::F32);
    w.data_as<float>()[0] = 1.f;
    w.data_as<float>()[1] = 1.f;

    Tensor y = rms_norm(x, w, 0.f);
    REQUIRE(y.at<float>({0, 0}) == Catch::Approx(y.at<float>({1, 0})).epsilon(1e-5));
    REQUIRE(y.at<float>({0, 1}) == Catch::Approx(y.at<float>({1, 1})).epsilon(1e-5));
}

TEST_CASE("rms_norm: zero vector with eps does not crash", "[ops][norm]") {
    Tensor x({4}, DType::F32);  // all zeros
    Tensor w({4}, DType::F32);
    for (size_t i = 0; i < 4; ++i) w.data_as<float>()[i] = 1.f;
    // Without eps this would divide by zero — must not crash or produce NaN.
    Tensor y = rms_norm(x, w, 1e-5f);
    for (size_t i = 0; i < 4; ++i)
        REQUIRE(std::isfinite(y.data_as<float>()[i]));
}
