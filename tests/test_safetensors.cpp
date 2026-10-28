#include <catch2/catch_test_macros.hpp>
#include "minillm/safetensors.hpp"

using namespace minillm;

// Minimal header that mirrors a real SmolLM2-135M safetensors file.
static const std::string SAMPLE_HEADER = R"({
    "__metadata__": {"format": "pt"},
    "embed_tokens.weight": {
        "dtype": "F32",
        "shape": [49152, 576],
        "data_offsets": [0, 113246208]
    },
    "layers.0.attn.q_proj.weight": {
        "dtype": "BF16",
        "shape": [576, 576],
        "data_offsets": [113246208, 113908224]
    }
})";

TEST_CASE("parse_safetensors_header: tensor count", "[safetensors]") {
    auto map = parse_safetensors_header(SAMPLE_HEADER);
    REQUIRE(map.size() == 2);
}

TEST_CASE("parse_safetensors_header: F32 shape and offsets", "[safetensors]") {
    auto& e = parse_safetensors_header(SAMPLE_HEADER).at("embed_tokens.weight");
    REQUIRE(e.dtype  == DType::F32);
    REQUIRE(e.shape  == std::vector<size_t>{49152, 576});
    REQUIRE(e.offset_start == 0);
    REQUIRE(e.offset_end   == 113246208);
}

TEST_CASE("parse_safetensors_header: BF16 dtype", "[safetensors]") {
    auto map = parse_safetensors_header(SAMPLE_HEADER);
    REQUIRE(map.at("layers.0.attn.q_proj.weight").dtype == DType::BF16);
}

TEST_CASE("parse_safetensors_header: unsupported dtype throws", "[safetensors]") {
    std::string bad = R"({"t":{"dtype":"F64","shape":[4],"data_offsets":[0,32]}})";
    REQUIRE_THROWS_AS(parse_safetensors_header(bad), std::runtime_error);
}
