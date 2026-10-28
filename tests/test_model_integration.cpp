#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>
#include "minillm/model.hpp"
#include "minillm/model_config.hpp"
#include "minillm/safetensors.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <numeric>

using namespace minillm;
namespace fs = std::filesystem;

static const fs::path PROJECT_ROOT  = MINILLM_PROJECT_ROOT;
static const fs::path MODEL_ST      = PROJECT_ROOT / "models/SmolLM2-135M/model.safetensors";
static const fs::path CONFIG_PATH   = PROJECT_ROOT / "models/SmolLM2-135M/config.json";
static const fs::path REF_LOGITS    = PROJECT_ROOT / "tools/reference_logits.json";

// Maximum absolute error we accept between C++ and PyTorch F32 logits.
// Small differences are expected from floating-point non-associativity.
static constexpr float kLogitTol = 2e-2f;

TEST_CASE("model forward: top-1 token matches PyTorch", "[model][integration]") {
    if (!fs::exists(MODEL_ST) || !fs::exists(CONFIG_PATH) || !fs::exists(REF_LOGITS))
        SKIP("Run download_model.py, export_weights.py, and gen_reference_logits.py first");

    // Load reference
    std::ifstream rf(REF_LOGITS);
    auto ref = nlohmann::json::parse(rf);
    const int expected_top1 = ref["top5_token_ids"][0].get<int>();
    auto input_ids_vec = ref["input_ids"].get<std::vector<int32_t>>();

    // Load model
    ModelConfig cfg = load_config(CONFIG_PATH.string());
    SafetensorsFile sf(MODEL_ST.string());
    ModelWeights w  = load_weights(sf, cfg);
    RopeCache rope  = build_rope_cache(cfg.max_position_embeddings,
                                        cfg.head_dim, cfg.rope_theta);

    // Forward pass
    Tensor logits = forward(w, cfg, rope, input_ids_vec);
    // logits: [seq_len, vocab_size] — take last token
    const size_t seq_len   = logits.shape()[0];
    const size_t vocab     = logits.shape()[1];
    const float* last_row  = logits.data_as<float>() + (seq_len - 1) * vocab;

    // Top-1
    const int cpp_top1 = static_cast<int>(
        std::max_element(last_row, last_row + vocab) - last_row);

    REQUIRE(cpp_top1 == expected_top1);
}

TEST_CASE("model forward: max abs error vs PyTorch within tolerance", "[model][integration]") {
    if (!fs::exists(MODEL_ST) || !fs::exists(CONFIG_PATH) || !fs::exists(REF_LOGITS))
        SKIP("Run download_model.py, export_weights.py, and gen_reference_logits.py first");

    std::ifstream rf(REF_LOGITS);
    auto ref = nlohmann::json::parse(rf);
    auto ref_logits    = ref["last_token_logits"].get<std::vector<float>>();
    auto input_ids_vec = ref["input_ids"].get<std::vector<int32_t>>();

    ModelConfig cfg = load_config(CONFIG_PATH.string());
    SafetensorsFile sf(MODEL_ST.string());
    ModelWeights w  = load_weights(sf, cfg);
    RopeCache rope  = build_rope_cache(cfg.max_position_embeddings,
                                        cfg.head_dim, cfg.rope_theta);

    Tensor logits = forward(w, cfg, rope, input_ids_vec);
    const size_t seq_len  = logits.shape()[0];
    const size_t vocab    = logits.shape()[1];
    const float* last_row = logits.data_as<float>() + (seq_len - 1) * vocab;

    float max_err = 0.f;
    for (size_t i = 0; i < vocab; ++i)
        max_err = std::max(max_err, std::abs(last_row[i] - ref_logits[i]));

    INFO("Max abs error vs PyTorch: " << max_err);
    REQUIRE(max_err < kLogitTol);
}
