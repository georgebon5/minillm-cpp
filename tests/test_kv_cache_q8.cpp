// Tests for the T26 INT8 KV-cache utility.
//
// Three contract checks:
//   1. append() stores one int8 row + one float scale per token, and the
//      per-row round-trip error is bounded by `scale`.
//   2. The public dequantize_row() matches dequantize_row_scalar() on
//      every platform (NEON ↔ scalar parity on AArch64; identity
//      elsewhere).
//   3. make_kvcache_q8(cfg) returns the expected shape and empty state.
//
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include "minillm/kv_cache_q8.hpp"
#include "minillm/model_config.hpp"

using namespace minillm;

namespace {

void rand_fill(std::vector<float>& v, float lo = -1.0f, float hi = 1.0f, unsigned seed = 42) {
    std::srand(seed);
    for (auto& x : v) {
        x = lo + (hi - lo) * (static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX));
    }
}

float max_abs_err(const std::vector<float>& a, const std::vector<float>& b) {
    REQUIRE(a.size() == b.size());
    float err = 0.0f;
    for (std::size_t i = 0; i < a.size(); ++i) {
        err = std::max(err, std::fabs(a[i] - b[i]));
    }
    return err;
}

} // namespace

// ─── make_kvcache_q8 ──────────────────────────────────────────────────────────

TEST_CASE("make_kvcache_q8: shape mirrors ModelConfig", "[kv_q8]") {
    ModelConfig cfg;
    cfg.num_hidden_layers   = 4;
    cfg.num_key_value_heads = 2;
    cfg.head_dim            = 32;

    const auto kv = make_kvcache_q8(cfg);
    REQUIRE(kv.layers.size() == 4);
    for (const auto& layer : kv.layers) {
        REQUIRE(layer.kv_dim == 64);
        REQUIRE(layer.seq_len == 0);
        REQUIRE(layer.k_data.empty());
        REQUIRE(layer.v_data.empty());
        REQUIRE(layer.k_scales.empty());
        REQUIRE(layer.v_scales.empty());
    }
}

// ─── append() + round-trip bound ─────────────────────────────────────────────

TEST_CASE("Layer::append stores one scale per token", "[kv_q8]") {
    KVCacheQ8::Layer layer;
    layer.kv_dim = 32;

    std::vector<float> k(32), v(32);
    rand_fill(k, -1.0f, 1.0f, 1);
    rand_fill(v, -2.0f, 2.0f, 2);
    layer.append(k.data(), v.data());

    REQUIRE(layer.seq_len == 1);
    REQUIRE(layer.k_data.size() == 32);
    REQUIRE(layer.v_data.size() == 32);
    REQUIRE(layer.k_scales.size() == 1);
    REQUIRE(layer.v_scales.size() == 1);
    REQUIRE(layer.k_scales[0] > 0.0f);
    REQUIRE(layer.v_scales[0] > 0.0f);
}

TEST_CASE("Layer::append round-trip error ≤ scale per row", "[kv_q8]") {
    KVCacheQ8::Layer layer;
    layer.kv_dim = 128;

    std::vector<float> k(128), v(128);
    rand_fill(k, -1.0f, 1.0f, 3);
    rand_fill(v, -1.0f, 1.0f, 4);
    layer.append(k.data(), v.data());

    std::vector<float> k_back(128), v_back(128);
    layer.dequantize_all_k(k_back.data());
    layer.dequantize_all_v(v_back.data());

    // Symmetric i8 quant: |x - x̂| ≤ scale for every lane (round-to-nearest).
    const float eps = 1e-6f;
    REQUIRE(max_abs_err(k, k_back) <= layer.k_scales[0] + eps);
    REQUIRE(max_abs_err(v, v_back) <= layer.v_scales[0] + eps);
}

TEST_CASE("Layer::append aggregates multiple tokens (independent scales)", "[kv_q8]") {
    KVCacheQ8::Layer layer;
    layer.kv_dim = 64;

    std::vector<float> row_small(64), row_large(64);
    rand_fill(row_small, -0.01f, 0.01f, 5);
    rand_fill(row_large, -10.0f, 10.0f, 6);

    layer.append(row_small.data(), row_small.data());
    layer.append(row_large.data(), row_large.data());

    REQUIRE(layer.seq_len == 2);
    REQUIRE(layer.k_scales.size() == 2);
    // Token with 1000× the dynamic range must get 1000× the scale (± rounding).
    REQUIRE(layer.k_scales[1] > 100.0f * layer.k_scales[0]);
}

// ─── Public dequantize_row() ↔ scalar parity ─────────────────────────────────

TEST_CASE("dequantize_row matches scalar reference (16-lane aligned)", "[kv_q8][neon]") {
    const std::size_t n = 64; // multiple of 16 → NEON path on AArch64
    std::vector<std::int8_t> data(n);
    std::srand(7);
    for (auto& x : data) x = static_cast<std::int8_t>((std::rand() % 255) - 127);

    const float scale = 0.0137f;
    std::vector<float> out_fast(n), out_ref(n);
    dequantize_row(scale, std::span<const std::int8_t>(data.data(), n), out_fast.data());
    dequantize_row_scalar(scale, std::span<const std::int8_t>(data.data(), n), out_ref.data());

    REQUIRE(max_abs_err(out_fast, out_ref) == Catch::Approx(0.0f).margin(1e-7f));
}

TEST_CASE("dequantize_row handles non-16-aligned tails", "[kv_q8][neon]") {
    const std::size_t n = 37; // 2 full tiles + 5-element tail
    std::vector<std::int8_t> data(n);
    std::srand(8);
    for (auto& x : data) x = static_cast<std::int8_t>((std::rand() % 255) - 127);

    const float scale = 0.0411f;
    std::vector<float> out_fast(n), out_ref(n);
    dequantize_row(scale, std::span<const std::int8_t>(data.data(), n), out_fast.data());
    dequantize_row_scalar(scale, std::span<const std::int8_t>(data.data(), n), out_ref.data());

    REQUIRE(max_abs_err(out_fast, out_ref) == Catch::Approx(0.0f).margin(1e-7f));
}

TEST_CASE("dequantize_row of an all-zero row returns all zeros", "[kv_q8]") {
    const std::size_t n = 32;
    std::vector<std::int8_t> data(n, 0);
    std::vector<float> out(n, 1.0f);
    dequantize_row(1.5f, std::span<const std::int8_t>(data.data(), n), out.data());
    for (float x : out) {
        REQUIRE(x == 0.0f);
    }
}

// ─── Footprint accounting (used by the capacity model in GreekKV T28) ────────

TEST_CASE("Layer::bytes reports int8 bodies + fp32 scales", "[kv_q8]") {
    KVCacheQ8::Layer layer;
    layer.kv_dim = 64;
    std::vector<float> row(64, 0.5f);
    for (int i = 0; i < 3; ++i) layer.append(row.data(), row.data());

    // 3 tokens * 64 bytes (K) + 3 * 64 (V) + 3 * 4 (K scales) + 3 * 4 (V scales)
    const std::size_t expected = 3 * 64 + 3 * 64 + 3 * 4 + 3 * 4;
    REQUIRE(layer.bytes() == expected);
}
