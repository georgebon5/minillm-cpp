#include <catch2/catch_test_macros.hpp>
#include "minillm/safetensors.hpp"
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using namespace minillm;

// ── Header parser tests ───────────────────────────────────────────────────────

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
    REQUIRE(e.dtype        == DType::F32);
    REQUIRE(e.shape        == std::vector<size_t>{49152, 576});
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

// ── SafetensorsFile mmap tests ────────────────────────────────────────────────

// Builds a minimal valid .safetensors byte blob for 4 F32 values {1,2,3,4}.
static std::vector<uint8_t> make_test_safetensors() {
    const std::string json =
        R"({"t":{"dtype":"F32","shape":[4],"data_offsets":[0,16]}})";

    uint64_t header_size = json.size();
    std::vector<float> data = {1.f, 2.f, 3.f, 4.f};

    std::vector<uint8_t> blob;
    blob.resize(8 + header_size + data.size() * sizeof(float));

    // Write header size as little-endian uint64.
    for (int i = 0; i < 8; ++i)
        blob[i] = static_cast<uint8_t>((header_size >> (8 * i)) & 0xff);

    std::memcpy(blob.data() + 8, json.data(), header_size);
    std::memcpy(blob.data() + 8 + header_size, data.data(), data.size() * sizeof(float));
    return blob;
}

// Returns the path of a temp file written with the given bytes.
// Caller is responsible for removing it.
static std::string write_temp_file(const std::vector<uint8_t>& bytes) {
    std::string path = std::string(std::tmpnam(nullptr)) + ".safetensors";
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
    return path;
}

TEST_CASE("SafetensorsFile: load and read tensor values", "[safetensors][mmap]") {
    auto blob = make_test_safetensors();
    auto path = write_temp_file(blob);

    {
        SafetensorsFile sf(path);
        REQUIRE(sf.has("t"));

        Tensor t = sf.tensor("t");
        REQUIRE(t.ndim()   == 1);
        REQUIRE(t.numel()  == 4);
        REQUIRE(t.dtype()  == DType::F32);
        REQUIRE(t.at<float>({0}) == 1.f);
        REQUIRE(t.at<float>({3}) == 4.f);
    }

    std::remove(path.c_str());
}

TEST_CASE("SafetensorsFile: unknown tensor throws", "[safetensors][mmap]") {
    auto blob = make_test_safetensors();
    auto path = write_temp_file(blob);

    {
        SafetensorsFile sf(path);
        REQUIRE_THROWS_AS(sf.tensor("nonexistent"), std::out_of_range);
    }

    std::remove(path.c_str());
}

TEST_CASE("SafetensorsFile: nonexistent file throws", "[safetensors][mmap]") {
    REQUIRE_THROWS_AS(SafetensorsFile("/tmp/does_not_exist_xyz.safetensors"),
                      std::runtime_error);
}
