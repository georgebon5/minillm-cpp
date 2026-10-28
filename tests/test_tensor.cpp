#include <catch2/catch_test_macros.hpp>
#include "minillm/tensor.hpp"

using namespace minillm;

TEST_CASE("Tensor shape and numel", "[tensor]") {
    Tensor t({2, 3, 4}, DType::F32);
    REQUIRE(t.ndim()   == 3);
    REQUIRE(t.numel()  == 24);
    REQUIRE(t.nbytes() == 24 * sizeof(float));
}

TEST_CASE("Tensor row-major strides", "[tensor]") {
    Tensor t({2, 3, 4}, DType::F32);
    REQUIRE(t.strides()[0] == 12);
    REQUIRE(t.strides()[1] == 4);
    REQUIRE(t.strides()[2] == 1);
}

TEST_CASE("Tensor data write and read", "[tensor]") {
    Tensor t({4}, DType::F32);
    float* p = t.data_as<float>();
    p[0] = 1.0f; p[1] = 2.0f; p[2] = 3.0f; p[3] = 4.0f;
    REQUIRE(t.data_as<float>()[2] == 3.0f);
}

TEST_CASE("Tensor I4 storage size", "[tensor]") {
    Tensor t5({5}, DType::I4);
    REQUIRE(t5.nbytes() == 3);   // ceil(5/2)
    Tensor t6({6}, DType::I4);
    REQUIRE(t6.nbytes() == 3);   // 6/2
}

TEST_CASE("Tensor scalar (0-d)", "[tensor]") {
    Tensor t({}, DType::F32);
    REQUIRE(t.numel() == 0);
    REQUIRE(t.ndim()  == 0);
}

TEST_CASE("Tensor at() computes correct offset", "[tensor]") {
    Tensor t({3, 4}, DType::F32);
    float* p = t.data_as<float>();
    // Fill with row*10 + col
    for (size_t r = 0; r < 3; ++r)
        for (size_t c = 0; c < 4; ++c)
            p[r * 4 + c] = float(r * 10 + c);

    REQUIRE(t.at<float>({0, 0}) == 0.f);
    REQUIRE(t.at<float>({1, 2}) == 12.f);
    REQUIRE(t.at<float>({2, 3}) == 23.f);
}

TEST_CASE("Tensor from_data shares memory (non-owning view)", "[tensor]") {
    std::vector<float> buf = {1.f, 2.f, 3.f, 4.f};
    Tensor view = Tensor::from_data(buf.data(), {2, 2}, DType::F32);

    REQUIRE_FALSE(view.owns_data());
    REQUIRE(view.at<float>({0, 1}) == 2.f);

    // Mutating the view mutates the original buffer.
    view.at<float>({1, 0}) = 99.f;
    REQUIRE(buf[2] == 99.f);
}
