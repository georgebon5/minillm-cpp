#include <catch2/catch_test_macros.hpp>
#include "minillm/model.hpp"
#include "minillm/model_config.hpp"
#include "minillm/safetensors.hpp"
#include "minillm/tokenizer.hpp"
#include <algorithm>
#include <filesystem>
#include <string>
#include <vector>

using namespace minillm;
namespace fs = std::filesystem;

static const fs::path PROJECT_ROOT = MINILLM_PROJECT_ROOT;
static const fs::path MODEL_ST     = PROJECT_ROOT / "models/SmolLM2-135M/model.safetensors";
static const fs::path CONFIG_PATH  = PROJECT_ROOT / "models/SmolLM2-135M/config.json";
static const fs::path TOK_PATH     = PROJECT_ROOT / "models/SmolLM2-135M/tokenizer.json";

// ── KVCache unit tests (no model files needed) ────────────────────────────────

TEST_CASE("make_kvcache: correct dimensions", "[kvcache]") {
    ModelConfig cfg;
    cfg.num_hidden_layers   = 4;
    cfg.num_attention_heads = 8;
    cfg.num_key_value_heads = 2;
    cfg.hidden_size         = 256;
    cfg.head_dim            = 256 / 8;   // 32
    cfg.kv_groups           = 8 / 2;

    KVCache kv = make_kvcache(cfg);
    REQUIRE(kv.layers.size() == 4);
    for (const auto& l : kv.layers) {
        REQUIRE(l.kv_dim  == 2u * 32u);   // n_kv_heads * head_dim = 64
        REQUIRE(l.seq_len == 0);
        REQUIRE(l.k.empty());
        REQUIRE(l.v.empty());
    }
}

TEST_CASE("KVCache::Layer::append grows correctly", "[kvcache]") {
    KVCache::Layer layer;
    layer.kv_dim = 4;

    std::vector<float> k1 = {1, 2, 3, 4};
    std::vector<float> v1 = {5, 6, 7, 8};
    layer.append(k1.data(), v1.data());

    REQUIRE(layer.seq_len == 1);
    REQUIRE(layer.k == std::vector<float>{1, 2, 3, 4});
    REQUIRE(layer.v == std::vector<float>{5, 6, 7, 8});

    std::vector<float> k2 = {9, 10, 11, 12};
    std::vector<float> v2 = {13, 14, 15, 16};
    layer.append(k2.data(), v2.data());

    REQUIRE(layer.seq_len == 2);
    REQUIRE(layer.k == (std::vector<float>{1,2,3,4, 9,10,11,12}));
    REQUIRE(layer.v == (std::vector<float>{5,6,7,8, 13,14,15,16}));
}

// ── Integration tests (SKIP if model files absent) ────────────────────────────

TEST_CASE("generate: output is non-empty and all ids in vocab", "[generation][integration]") {
    if (!fs::exists(MODEL_ST) || !fs::exists(CONFIG_PATH) || !fs::exists(TOK_PATH))
        SKIP("Run download_model.py first");

    ModelConfig   cfg  = load_config(CONFIG_PATH.string());
    SafetensorsFile sf(MODEL_ST.string());
    ModelWeights  w    = load_weights(sf, cfg);
    RopeCache     rope = build_rope_cache(cfg.max_position_embeddings,
                                          cfg.head_dim, cfg.rope_theta);
    Tokenizer     tok  = load_tokenizer(TOK_PATH.string());

    auto prompt_ids = encode(tok, "The capital of France is");
    auto gen_ids    = generate(w, cfg, rope, prompt_ids, /*max_new_tokens=*/8,
                               tok.eos_id);

    REQUIRE(!gen_ids.empty());
    for (int32_t id : gen_ids) {
        REQUIRE(id >= 0);
        REQUIRE(id < tok.vocab_size());
    }
}

TEST_CASE("generate: greedy first token matches forward() top-1", "[generation][integration]") {
    if (!fs::exists(MODEL_ST) || !fs::exists(CONFIG_PATH) || !fs::exists(TOK_PATH))
        SKIP("Run download_model.py first");

    ModelConfig   cfg  = load_config(CONFIG_PATH.string());
    SafetensorsFile sf(MODEL_ST.string());
    ModelWeights  w    = load_weights(sf, cfg);
    RopeCache     rope = build_rope_cache(cfg.max_position_embeddings,
                                          cfg.head_dim, cfg.rope_theta);
    Tokenizer     tok  = load_tokenizer(TOK_PATH.string());

    auto prompt_ids = encode(tok, "The capital of France is");

    // Reference: full-sequence forward → top-1 of last position
    Tensor ref_logits = forward(w, cfg, rope, prompt_ids);
    const size_t vocab    = ref_logits.shape()[1];
    const size_t seq_len  = ref_logits.shape()[0];
    const float* last_row = ref_logits.data_as<float>() + (seq_len - 1) * vocab;
    const int32_t ref_top1 = static_cast<int32_t>(
        std::max_element(last_row, last_row + vocab) - last_row);

    // generate() with max_new_tokens=1
    auto gen = generate(w, cfg, rope, prompt_ids, 1, tok.eos_id);
    REQUIRE(!gen.empty());
    REQUIRE(gen[0] == ref_top1);
}
