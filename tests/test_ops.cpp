#include <catch2/catch_test_macros.hpp>
#include "minillm/ops.hpp"
#include <array>

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
