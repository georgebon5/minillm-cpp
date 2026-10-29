#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "minillm/model_config.hpp"

using namespace minillm;

static const char* SMOLLM2_JSON = R"({
    "vocab_size": 49152,
    "hidden_size": 576,
    "intermediate_size": 1536,
    "num_hidden_layers": 30,
    "num_attention_heads": 9,
    "num_key_value_heads": 3,
    "max_position_embeddings": 2048,
    "rms_norm_eps": 1e-5,
    "rope_theta": 10000.0,
    "bos_token_id": 1,
    "eos_token_id": 2
})";

TEST_CASE("parse_config: SmolLM2-135M dimensions", "[config]") {
    auto c = parse_config(SMOLLM2_JSON);
    REQUIRE(c.vocab_size              == 49152);
    REQUIRE(c.hidden_size             == 576);
    REQUIRE(c.intermediate_size       == 1536);
    REQUIRE(c.num_hidden_layers       == 30);
    REQUIRE(c.num_attention_heads     == 9);
    REQUIRE(c.num_key_value_heads     == 3);
    REQUIRE(c.max_position_embeddings == 2048);
}

TEST_CASE("parse_config: derived quantities", "[config]") {
    auto c = parse_config(SMOLLM2_JSON);
    REQUIRE(c.head_dim  == 64);   // 576 / 9
    REQUIRE(c.kv_groups == 3);    // 9  / 3
}

TEST_CASE("parse_config: hyperparameters", "[config]") {
    auto c = parse_config(SMOLLM2_JSON);
    REQUIRE(c.rms_norm_eps == Catch::Approx(1e-5f));
    REQUIRE(c.rope_theta   == Catch::Approx(10000.f));
    REQUIRE(c.bos_token_id == 1);
    REQUIRE(c.eos_token_id == 2);
}

TEST_CASE("parse_config: bad head divisibility throws", "[config]") {
    // 9 attention heads but 4 KV heads → 9 % 4 != 0
    std::string bad = R"({
        "vocab_size":49152,"hidden_size":576,"intermediate_size":1536,
        "num_hidden_layers":30,"num_attention_heads":9,"num_key_value_heads":4,
        "max_position_embeddings":2048,"rms_norm_eps":1e-5,
        "bos_token_id":1,"eos_token_id":2
    })";
    REQUIRE_THROWS_AS(parse_config(bad), std::runtime_error);
}

TEST_CASE("parse_config: bad hidden/head divisibility throws", "[config]") {
    // hidden_size=577 is not divisible by 9
    std::string bad = R"({
        "vocab_size":49152,"hidden_size":577,"intermediate_size":1536,
        "num_hidden_layers":30,"num_attention_heads":9,"num_key_value_heads":3,
        "max_position_embeddings":2048,"rms_norm_eps":1e-5,
        "bos_token_id":1,"eos_token_id":2
    })";
    REQUIRE_THROWS_AS(parse_config(bad), std::runtime_error);
}
