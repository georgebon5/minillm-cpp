#include <catch2/catch_test_macros.hpp>
#include "minillm/tensor.hpp"

using namespace minillm;

TEST_CASE("Tensor shape and numel", "[tensor]") {
    Tensor t({2, 3, 4}, DType::F32);
    REQUIRE(t.ndim()  == 3);
    REQUIRE(t.numel() == 24);
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
    REQUIRE(t.numel()  == 0);
    REQUIRE(t.ndim()   == 0);
}
