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

// ── Linear ────────────────────────────────────────────────────────────────────

TEST_CASE("linear: output shape", "[ops][linear]") {
    Tensor x({3, 4}, DType::F32);
    Tensor W({5, 4}, DType::F32);
    Tensor y = linear(x, W);
    REQUIRE(y.shape() == std::vector<size_t>{3, 5});
}

TEST_CASE("linear: identity weight", "[ops][linear]") {
    // W = I_2 → y = x
    Tensor x({2, 2}, DType::F32);
    float* xp = x.data_as<float>();
    xp[0] = 1.f; xp[1] = 2.f;
    xp[2] = 3.f; xp[3] = 4.f;

    Tensor W({2, 2}, DType::F32);
    float* Wp = W.data_as<float>();
    Wp[0] = 1.f; Wp[1] = 0.f;
    Wp[2] = 0.f; Wp[3] = 1.f;

    Tensor y = linear(x, W);
    REQUIRE(y.at<float>({0, 0}) == Catch::Approx(1.f));
    REQUIRE(y.at<float>({0, 1}) == Catch::Approx(2.f));
    REQUIRE(y.at<float>({1, 0}) == Catch::Approx(3.f));
    REQUIRE(y.at<float>({1, 1}) == Catch::Approx(4.f));
}

TEST_CASE("linear: known matrix multiply", "[ops][linear]") {
    // x = [[1,2,3]]  W = [[1,1,1],[2,2,2]]
    // y[0,0] = 1+2+3 = 6,  y[0,1] = 2+4+6 = 12
    Tensor x({1, 3}, DType::F32);
    x.data_as<float>()[0] = 1.f;
    x.data_as<float>()[1] = 2.f;
    x.data_as<float>()[2] = 3.f;

    Tensor W({2, 3}, DType::F32);
    for (int i = 0; i < 3; ++i) W.data_as<float>()[i]     = 1.f;
    for (int i = 0; i < 3; ++i) W.data_as<float>()[3 + i] = 2.f;

    Tensor y = linear(x, W);
    REQUIRE(y.at<float>({0, 0}) == Catch::Approx(6.f));
    REQUIRE(y.at<float>({0, 1}) == Catch::Approx(12.f));
}

TEST_CASE("linear: bias is added", "[ops][linear]") {

    // Same x, W as above + bias [10, 20]
    Tensor x({1, 3}, DType::F32);
    x.data_as<float>()[0] = 1.f;
    x.data_as<float>()[1] = 2.f;
    x.data_as<float>()[2] = 3.f;

    Tensor W({2, 3}, DType::F32);
    for (int i = 0; i < 3; ++i) W.data_as<float>()[i]     = 1.f;
    for (int i = 0; i < 3; ++i) W.data_as<float>()[3 + i] = 2.f;

    Tensor b({2}, DType::F32);
    b.data_as<float>()[0] = 10.f;
    b.data_as<float>()[1] = 20.f;

    Tensor y = linear(x, W, &b);
    REQUIRE(y.at<float>({0, 0}) == Catch::Approx(16.f));
    REQUIRE(y.at<float>({0, 1}) == Catch::Approx(32.f));
}

// ── RoPE cache ────────────────────────────────────────────────────────────────

TEST_CASE("build_rope_cache: output shape", "[ops][rope]") {
    auto cache = build_rope_cache(128, 64, 10000.f);
    REQUIRE(cache.cos.shape() == std::vector<size_t>{128, 32});
    REQUIRE(cache.sin.shape() == std::vector<size_t>{128, 32});
}

TEST_CASE("build_rope_cache: position 0 has cos=1 sin=0", "[ops][rope]") {
    auto cache = build_rope_cache(8, 4, 10000.f);
    // At m=0: angle = 0 * θ_i = 0 → cos=1, sin=0 for all i
    for (size_t i = 0; i < 2; ++i) {
        REQUIRE(cache.cos.at<float>({0, i}) == Catch::Approx(1.f).epsilon(1e-6));
        REQUIRE(cache.sin.at<float>({0, i}) == Catch::Approx(0.f).margin(1e-6));
    }
}

TEST_CASE("build_rope_cache: cos²+sin²=1 for all entries", "[ops][rope]") {
    auto cache = build_rope_cache(32, 8, 10000.f);
    const float* cp = cache.cos.data_as<float>();
    const float* sp = cache.sin.data_as<float>();
    for (size_t k = 0; k < cache.cos.numel(); ++k)
        REQUIRE(cp[k] * cp[k] + sp[k] * sp[k] == Catch::Approx(1.f).epsilon(1e-5));
}

TEST_CASE("build_rope_cache: first freq θ₀=1 at position 1", "[ops][rope]") {
    // θ_0 = 1/(10000^(0/d)) = 1.0 → angle at m=1 is 1.0 rad
    auto cache = build_rope_cache(4, 4, 10000.f);
    REQUIRE(cache.cos.at<float>({1, 0}) == Catch::Approx(std::cos(1.f)).epsilon(1e-5));
    REQUIRE(cache.sin.at<float>({1, 0}) == Catch::Approx(std::sin(1.f)).epsilon(1e-5));
}

TEST_CASE("build_rope_cache: SmolLM2 dimensions are valid", "[ops][rope]") {
    // Verify no crash and correct shape for the actual model config.
    auto cache = build_rope_cache(2048, 64, 10000.f);
    REQUIRE(cache.cos.numel() == 2048 * 32);
    REQUIRE(cache.sin.numel() == 2048 * 32);
}

// ── RoPE apply ────────────────────────────────────────────────────────────────

TEST_CASE("rope_apply: position 0 is a no-op", "[ops][rope]") {
    // At pos=0: cos=1, sin=0 → x' = x
    auto cache = build_rope_cache(4, 4, 10000.f);

    Tensor x({1, 4}, DType::F32);  // seq_len=1, n_heads=1, head_dim=4
    float* p = x.data_as<float>();
    p[0] = 3.f; p[1] = 4.f; p[2] = 1.f; p[3] = 2.f;

    rope_apply(x, cache, 0);

    REQUIRE(x.at<float>({0, 0}) == Catch::Approx(3.f).epsilon(1e-6));
    REQUIRE(x.at<float>({0, 1}) == Catch::Approx(4.f).epsilon(1e-6));
}

TEST_CASE("rope_apply: known rotation at position 1 (rope_theta=1)", "[ops][rope]") {
    // With rope_theta=1: θ_0 = 1/(1^0) = 1.0 rad at m=1.
    // x = [1, 0]  →  x' = [cos(1), sin(1)]
    auto cache = build_rope_cache(4, 2, 1.f);  // head_dim=2, half=1

    Tensor x({1, 2}, DType::F32);
    x.data_as<float>()[0] = 1.f;
    x.data_as<float>()[1] = 0.f;

    rope_apply(x, cache, /*pos_offset=*/1);

    REQUIRE(x.at<float>({0, 0}) == Catch::Approx(std::cos(1.f)).epsilon(1e-5));
    REQUIRE(x.at<float>({0, 1}) == Catch::Approx(std::sin(1.f)).epsilon(1e-5));
}

TEST_CASE("rope_apply: rotation preserves vector norm", "[ops][rope]") {
    auto cache = build_rope_cache(16, 8, 10000.f);

    Tensor x({4, 8}, DType::F32);  // 4 tokens, 1 head, head_dim=8
    float* p = x.data_as<float>();
    for (size_t i = 0; i < 32; ++i)
        p[i] = static_cast<float>(i + 1);

    // Compute norms before rotation.
    std::vector<float> norms_before(4);
    for (int t = 0; t < 4; ++t) {
        float n = 0.f;
        for (int d = 0; d < 8; ++d) n += p[t*8+d] * p[t*8+d];
        norms_before[t] = n;
    }

    rope_apply(x, cache, 0);

    for (int t = 0; t < 4; ++t) {
        float n = 0.f;
        for (int d = 0; d < 8; ++d) n += x.data_as<float>()[t*8+d] * x.data_as<float>()[t*8+d];
        REQUIRE(n == Catch::Approx(norms_before[t]).epsilon(1e-4));
    }
}

TEST_CASE("rope_apply: pos_offset shifts positions correctly", "[ops][rope]") {
    // Applying with pos_offset=2 should give same result as taking row 2 from cache.
    auto cache = build_rope_cache(8, 2, 1.f);

    Tensor x1({1, 2}, DType::F32);
    x1.data_as<float>()[0] = 1.f; x1.data_as<float>()[1] = 0.f;
    rope_apply(x1, cache, 2);

    // Direct check: angle = 2*1.0 = 2 rad
    REQUIRE(x1.at<float>({0, 0}) == Catch::Approx(std::cos(2.f)).epsilon(1e-5));
    REQUIRE(x1.at<float>({0, 1}) == Catch::Approx(std::sin(2.f)).epsilon(1e-5));
}

// ── Softmax ───────────────────────────────────────────────────────────────────

TEST_CASE("softmax_inplace: sums to 1", "[ops][softmax]") {
    std::vector<float> v = {1.f, 2.f, 3.f, 4.f};
    softmax_inplace(v.data(), v.size());
    float sum = 0.f;
    for (float x : v) sum += x;
    REQUIRE(sum == Catch::Approx(1.f).epsilon(1e-6));
}

TEST_CASE("softmax_inplace: uniform input gives uniform output", "[ops][softmax]") {
    std::vector<float> v = {1.f, 1.f, 1.f, 1.f};
    softmax_inplace(v.data(), v.size());
    for (float x : v)
        REQUIRE(x == Catch::Approx(0.25f).epsilon(1e-6));
}

TEST_CASE("softmax_inplace: large values do not overflow", "[ops][softmax]") {
    std::vector<float> v = {1000.f, 1001.f, 1002.f};
    softmax_inplace(v.data(), v.size());
    float sum = 0.f;
    for (float x : v) { REQUIRE(std::isfinite(x)); sum += x; }
    REQUIRE(sum == Catch::Approx(1.f).epsilon(1e-6));
}

// ── Scaled dot-product attention ──────────────────────────────────────────────

// Helper: make a [n, d] tensor filled with a constant.
static Tensor make_const_tensor(size_t n, size_t d, float val) {
    Tensor t({n, d}, DType::F32);
    for (size_t i = 0; i < n * d; ++i) t.data_as<float>()[i] = val;
    return t;
}

TEST_CASE("sdp_attention: output shape", "[ops][attention]") {
    auto q = make_const_tensor(3, 4, 0.f);
    auto k = make_const_tensor(3, 4, 0.f);
    auto v = make_const_tensor(3, 4, 1.f);
    auto out = scaled_dot_product_attention(q, k, v);
    REQUIRE(out.shape() == std::vector<size_t>{3, 4});
}

TEST_CASE("sdp_attention: uniform scores → output is mean of V rows", "[ops][attention]") {
    // Q=K=zero → all scores=0 → softmax gives uniform weights 1/N
    // output[i] = (1/N) * sum_j V[j]
    auto q = make_const_tensor(2, 2, 0.f);
    auto k = make_const_tensor(2, 2, 0.f);

    Tensor v({2, 2}, DType::F32);
    v.data_as<float>()[0] = 1.f; v.data_as<float>()[1] = 2.f;  // row 0
    v.data_as<float>()[2] = 3.f; v.data_as<float>()[3] = 4.f;  // row 1

    auto out = scaled_dot_product_attention(q, k, v);

    // Expected: 0.5*[1,2] + 0.5*[3,4] = [2, 3]
    REQUIRE(out.at<float>({0, 0}) == Catch::Approx(2.f).epsilon(1e-5));
    REQUIRE(out.at<float>({0, 1}) == Catch::Approx(3.f).epsilon(1e-5));
    REQUIRE(out.at<float>({1, 0}) == Catch::Approx(2.f).epsilon(1e-5));
}

TEST_CASE("sdp_attention: single token output equals V", "[ops][attention]") {
    // seq_len=1: softmax([score]) = [1.0] → output = 1.0 * V[0]
    Tensor q({1, 4}, DType::F32); q.data_as<float>()[0] = 1.f;
    Tensor k({1, 4}, DType::F32); k.data_as<float>()[0] = 1.f;
    Tensor v({1, 4}, DType::F32);
    for (int i = 0; i < 4; ++i) v.data_as<float>()[i] = float(i + 1);

    auto out = scaled_dot_product_attention(q, k, v);
    for (int i = 0; i < 4; ++i)
        REQUIRE(out.data_as<float>()[i] == Catch::Approx(float(i + 1)).epsilon(1e-5));
}

// ── Causal mask ───────────────────────────────────────────────────────────────

TEST_CASE("sdp_attention: causal mask — token 0 only sees itself", "[ops][causal]") {
    // Q=K=zero → uniform raw scores. With causal mask:
    // Row 0: attend only to j=0 → output[0] = V[0]
    auto q = make_const_tensor(3, 3, 0.f);
    auto k = make_const_tensor(3, 3, 0.f);

    // V = identity-like rows for easy verification
    Tensor v({3, 3}, DType::F32);
    float* vp = v.data_as<float>();
    vp[0]=1.f; vp[1]=0.f; vp[2]=0.f;  // row 0
    vp[3]=0.f; vp[4]=1.f; vp[5]=0.f;  // row 1
    vp[6]=0.f; vp[7]=0.f; vp[8]=1.f;  // row 2

    auto out = scaled_dot_product_attention(q, k, v, /*causal=*/true);

    // Token 0 attends only to position 0 → output = V[0]
    REQUIRE(out.at<float>({0, 0}) == Catch::Approx(1.f).epsilon(1e-5));
    REQUIRE(out.at<float>({0, 1}) == Catch::Approx(0.f).margin(1e-5));
    REQUIRE(out.at<float>({0, 2}) == Catch::Approx(0.f).margin(1e-5));
}

TEST_CASE("sdp_attention: causal mask — token 1 averages positions 0 and 1", "[ops][causal]") {
    auto q = make_const_tensor(3, 3, 0.f);
    auto k = make_const_tensor(3, 3, 0.f);
    Tensor v({3, 3}, DType::F32);
    float* vp = v.data_as<float>();
    vp[0]=1.f; vp[1]=0.f; vp[2]=0.f;
    vp[3]=0.f; vp[4]=1.f; vp[5]=0.f;
    vp[6]=0.f; vp[7]=0.f; vp[8]=1.f;

    auto out = scaled_dot_product_attention(q, k, v, /*causal=*/true);

    // Token 1 attends to positions 0 and 1 with equal weight 0.5
    REQUIRE(out.at<float>({1, 0}) == Catch::Approx(0.5f).epsilon(1e-5));
    REQUIRE(out.at<float>({1, 1}) == Catch::Approx(0.5f).epsilon(1e-5));
    REQUIRE(out.at<float>({1, 2}) == Catch::Approx(0.f).margin(1e-5));
}

TEST_CASE("sdp_attention: causal=false matches causal=true for seq_len=1", "[ops][causal]") {
    Tensor q({1, 4}, DType::F32);
    Tensor k({1, 4}, DType::F32);
    Tensor v({1, 4}, DType::F32);
    for (int i = 0; i < 4; ++i) {
        q.data_as<float>()[i] = float(i);
        k.data_as<float>()[i] = float(i);
        v.data_as<float>()[i] = float(i + 1);
    }
    auto out_nc = scaled_dot_product_attention(q, k, v, false);
    auto out_c  = scaled_dot_product_attention(q, k, v, true);
    for (int i = 0; i < 4; ++i)
        REQUIRE(out_nc.data_as<float>()[i] ==
                Catch::Approx(out_c.data_as<float>()[i]).epsilon(1e-6));
}

// ── GQA attention ─────────────────────────────────────────────────────────────

// Build a tiny GQA setup:  hidden=4, n_heads=2, n_kv_heads=1, head_dim=2
static void make_gqa_weights(Tensor& Wq, Tensor& Wk, Tensor& Wv, Tensor& Wo) {
    // Wq: [4, 4], Wk: [2, 4], Wv: [2, 4], Wo: [4, 4]
    Wq = Tensor({4, 4}, DType::F32);
    Wk = Tensor({2, 4}, DType::F32);
    Wv = Tensor({2, 4}, DType::F32);
    Wo = Tensor({4, 4}, DType::F32);
    // Identity-like initialisation: set diagonal to 1
    auto set_eye = [](Tensor& t) {
        size_t n = std::min(t.shape()[0], t.shape()[1]);
        for (size_t i = 0; i < n; ++i)
            t.data_as<float>()[i * t.shape()[1] + i] = 1.f;
    };
    set_eye(Wq); set_eye(Wk); set_eye(Wv); set_eye(Wo);
}

TEST_CASE("gqa_attention: output shape", "[ops][gqa]") {
    Tensor Wq, Wk, Wv, Wo;
    make_gqa_weights(Wq, Wk, Wv, Wo);
    auto rope = build_rope_cache(16, /*head_dim=*/2, 10000.f);

    Tensor x({3, 4}, DType::F32);
    for (size_t i = 0; i < 12; ++i) x.data_as<float>()[i] = float(i) * 0.1f;

    auto out = gqa_attention(x, Wq, Wk, Wv, Wo, rope, /*n_heads=*/2, /*n_kv_heads=*/1);
    REQUIRE(out.shape() == std::vector<size_t>{3, 4});
}

TEST_CASE("gqa_attention: output is finite", "[ops][gqa]") {
    Tensor Wq, Wk, Wv, Wo;
    make_gqa_weights(Wq, Wk, Wv, Wo);
    auto rope = build_rope_cache(16, 2, 10000.f);

    Tensor x({4, 4}, DType::F32);
    for (size_t i = 0; i < 16; ++i) x.data_as<float>()[i] = float(i + 1) * 0.01f;

    auto out = gqa_attention(x, Wq, Wk, Wv, Wo, rope, 2, 1);
    for (size_t i = 0; i < out.numel(); ++i)
        REQUIRE(std::isfinite(out.data_as<float>()[i]));
}

TEST_CASE("gqa_attention: single token output is finite", "[ops][gqa]") {
    Tensor Wq, Wk, Wv, Wo;
    make_gqa_weights(Wq, Wk, Wv, Wo);
    auto rope = build_rope_cache(16, 2, 10000.f);

    Tensor x({1, 4}, DType::F32);
    x.data_as<float>()[0] = 1.f; x.data_as<float>()[1] = 0.f;
    x.data_as<float>()[2] = 0.f; x.data_as<float>()[3] = 1.f;

    auto out = gqa_attention(x, Wq, Wk, Wv, Wo, rope, 2, 1, /*pos_offset=*/5);
    REQUIRE(out.shape() == std::vector<size_t>{1, 4});
    for (size_t i = 0; i < out.numel(); ++i)
        REQUIRE(std::isfinite(out.data_as<float>()[i]));
}
