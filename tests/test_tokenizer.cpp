#include <catch2/catch_test_macros.hpp>
#include "minillm/tokenizer.hpp"
#include <string>

using namespace minillm;

// Minimal tokenizer.json fixture — two real tokens + one added special token.
static const std::string kMinimalJson = R"({
  "model": {
    "type": "BPE",
    "vocab": {
      "<unk>": 0,
      "<s>":   1,
      "</s>":  2,
      "he":    3,
      "llo":   4,
      "Ġworld": 5
    },
    "merges": [
      "h e",
      "he llo"
    ]
  },
  "added_tokens": [
    {"id": 0, "content": "<unk>",  "special": true},
    {"id": 1, "content": "<s>",    "special": true},
    {"id": 2, "content": "</s>",   "special": true}
  ]
})";

TEST_CASE("parse_tokenizer: vocab size", "[tokenizer]") {
    Tokenizer tok = parse_tokenizer(kMinimalJson);
    REQUIRE(tok.vocab_size() == 6);
}

TEST_CASE("parse_tokenizer: token_to_id round-trip", "[tokenizer]") {
    Tokenizer tok = parse_tokenizer(kMinimalJson);
    REQUIRE(tok.token_to_id.at("he")      == 3);
    REQUIRE(tok.token_to_id.at("llo")     == 4);
    REQUIRE(tok.token_to_id.at("Ġworld") == 5);
}

TEST_CASE("parse_tokenizer: id_to_token round-trip", "[tokenizer]") {
    Tokenizer tok = parse_tokenizer(kMinimalJson);
    REQUIRE(tok.id_to_token[3] == "he");
    REQUIRE(tok.id_to_token[4] == "llo");
    REQUIRE(tok.id_to_token[5] == "Ġworld");
}

TEST_CASE("parse_tokenizer: merge ranks", "[tokenizer]") {
    Tokenizer tok = parse_tokenizer(kMinimalJson);
    REQUIRE(tok.merge_ranks.at("h e")    == 0);   // highest priority
    REQUIRE(tok.merge_ranks.at("he llo") == 1);
    REQUIRE(tok.merge_ranks.size() == 2);
}

TEST_CASE("parse_tokenizer: special token ids", "[tokenizer]") {
    Tokenizer tok = parse_tokenizer(kMinimalJson);
    REQUIRE(tok.unk_id == 0);
    REQUIRE(tok.bos_id == 1);
    REQUIRE(tok.eos_id == 2);
}
