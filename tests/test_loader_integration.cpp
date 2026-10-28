#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>
#include "minillm/safetensors.hpp"
#include "minillm/model_config.hpp"
#include <cmath>
#include <filesystem>
#include <fstream>

using namespace minillm;
namespace fs = std::filesystem;

// Paths are relative to the repo root, injected at compile time.
static const fs::path PROJECT_ROOT = MINILLM_PROJECT_ROOT;
static const fs::path MODEL_ST   = PROJECT_ROOT / "models/SmolLM2-135M/model.safetensors";
static const fs::path CONFIG_JSON = PROJECT_ROOT / "models/SmolLM2-135M/config.json";
static const fs::path REF_JSON    = PROJECT_ROOT / "tools/reference_tensors.json";

// Max absolute error allowed between C++ and Python F32 values.
// BF16→F32 is a deterministic bit operation so we expect exact equality,
// but allow a tiny epsilon for any platform float rounding.
static constexpr float kTol = 1e-6f;

TEST_CASE("Loader: all tensor shapes match Python reference", "[integration]") {
    if (!fs::exists(MODEL_ST) || !fs::exists(REF_JSON))
        SKIP("Run download_model.py and export_weights.py to generate test fixtures");

    std::ifstream rf(REF_JSON);
    REQUIRE(rf.is_open());
    auto ref = nlohmann::json::parse(rf);

    SafetensorsFile sf(MODEL_ST.string());

    for (auto& [name, info] : ref.items()) {
        CAPTURE(name);
        REQUIRE(sf.has(name));

        auto expected = info["shape"].get<std::vector<size_t>>();
        Tensor t = sf.tensor(name);   // non-owning, original dtype
        REQUIRE(t.shape() == expected);
    }
}

TEST_CASE("Loader: first 4 F32 values match Python reference", "[integration]") {
    if (!fs::exists(MODEL_ST) || !fs::exists(REF_JSON))
        SKIP("Run download_model.py and export_weights.py to generate test fixtures");

    std::ifstream rf(REF_JSON);
    auto ref = nlohmann::json::parse(rf);

    SafetensorsFile sf(MODEL_ST.string());

    for (auto& [name, info] : ref.items()) {
        CAPTURE(name);

        auto expected_vals = info["first_4"].get<std::vector<float>>();
        Tensor t = sf.tensor_f32(name);   // owning F32, BF16 converted
        const float* data = t.data_as<float>();

        for (size_t i = 0; i < expected_vals.size() && i < t.numel(); ++i) {
            CAPTURE(i);
            CHECK(std::abs(data[i] - expected_vals[i]) <= kTol);
        }
    }
}

TEST_CASE("Loader: config.json dimensions are consistent with tensor shapes", "[integration]") {
    if (!fs::exists(MODEL_ST) || !fs::exists(CONFIG_JSON))
        SKIP("Run download_model.py to generate test fixtures");

    ModelConfig cfg = load_config(CONFIG_JSON.string());
    SafetensorsFile sf(MODEL_ST.string());

    // Embedding table: [vocab_size, hidden_size]
    {
        Tensor emb = sf.tensor("model.embed_tokens.weight");
        REQUIRE(emb.shape()[0] == static_cast<size_t>(cfg.vocab_size));
        REQUIRE(emb.shape()[1] == static_cast<size_t>(cfg.hidden_size));
    }

    // First layer RMSNorm weight: [hidden_size]
    {
        Tensor norm = sf.tensor("model.layers.0.input_layernorm.weight");
        REQUIRE(norm.shape()[0] == static_cast<size_t>(cfg.hidden_size));
    }

    // First layer Q projection: [num_heads * head_dim, hidden_size]
    {
        Tensor q = sf.tensor("model.layers.0.self_attn.q_proj.weight");
        REQUIRE(q.shape()[0] == static_cast<size_t>(cfg.num_attention_heads * cfg.head_dim));
        REQUIRE(q.shape()[1] == static_cast<size_t>(cfg.hidden_size));
    }
}
