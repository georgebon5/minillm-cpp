#include <catch2/catch_test_macros.hpp>
#include "minillm/tokenizer.hpp"
#include <filesystem>
#include <string>

using namespace minillm;
namespace fs = std::filesystem;
static const fs::path PROJECT_ROOT = MINILLM_PROJECT_ROOT;

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

// ── encode tests ─────────────────────────────────────────────────────────────
// Fixture with individual chars + step-by-step merges so we can predict ids.
// Ġ is the JSON unicode escape for U+0120 (Ġ = GPT-2 space prefix).
// Using the escape rather than the literal character avoids editor encoding
// ambiguity; nlohmann::json parses it to \xC4\xA0 at runtime.
static const std::string kEncodeJson =
    "{\n"
    "  \"model\": {\n"
    "    \"type\": \"BPE\",\n"
    "    \"vocab\": {\n"
    "      \"<unk>\": 0,  \"<s>\": 1,  \"</s>\": 2,\n"
    "      \"h\": 3,  \"i\": 4,  \"hi\": 5,\n"
    "      \"t\": 6,  \"e\": 7,  \"r\": 8,\n"
    "      \"\\u0120\": 9,\n"
    "      \"\\u0120t\": 10,  \"\\u0120th\": 11,  \"\\u0120the\": 12,\n"
    "      \"\\u0120ther\": 13,  \"\\u0120there\": 14\n"
    "    },\n"
    "    \"merges\": [\n"
    "      \"h i\",\n"
    "      \"\\u0120 t\",\n"
    "      \"\\u0120t h\",\n"
    "      \"\\u0120th e\",\n"
    "      \"\\u0120the r\",\n"
    "      \"\\u0120ther e\"\n"
    "    ]\n"
    "  },\n"
    "  \"added_tokens\": [\n"
    "    {\"id\": 0, \"content\": \"<unk>\",  \"special\": true},\n"
    "    {\"id\": 1, \"content\": \"<s>\",    \"special\": true},\n"
    "    {\"id\": 2, \"content\": \"</s>\",   \"special\": true}\n"
    "  ]\n"
    "}";

TEST_CASE("encode: empty string", "[tokenizer][encode]") {
    Tokenizer tok = parse_tokenizer(kEncodeJson);
    REQUIRE(encode(tok, "").empty());
}

TEST_CASE("encode: single merged token", "[tokenizer][encode]") {
    // "hi" → pretokenize → ["hi"], BPE: ["h","i"] → merge "h i" → ["hi"] → id 5
    Tokenizer tok = parse_tokenizer(kEncodeJson);
    auto ids = encode(tok, "hi");
    REQUIRE(ids == std::vector<int32_t>{5});
}

TEST_CASE("encode: space-prefixed word merges correctly", "[tokenizer][encode]") {
    // " there" → pretokenize → ["Ġthere"]
    // BPE: Ġ t h e r e  →  (step-by-step merges)  →  ["Ġthere"] → id 14
    Tokenizer tok = parse_tokenizer(kEncodeJson);
    auto ids = encode(tok, " there");
    REQUIRE(ids == std::vector<int32_t>{14});
}

TEST_CASE("encode: two words produce two token sequences", "[tokenizer][encode]") {
    // "hi there" → ["hi"] ++ ["Ġthere"] → [5, 14]
    Tokenizer tok = parse_tokenizer(kEncodeJson);
    auto ids = encode(tok, "hi there");
    REQUIRE(ids == std::vector<int32_t>{5, 14});
}

TEST_CASE("encode: unknown chars map to unk_id", "[tokenizer][encode]") {
    // "x" has no entry in this vocab after BPE → unk
    Tokenizer tok = parse_tokenizer(kEncodeJson);
    auto ids = encode(tok, "x");
    REQUIRE(ids.size() == 1);
    REQUIRE(ids[0] == tok.unk_id);
}

// ── integration test (SKIP if tokenizer.json not present) ────────────────────
TEST_CASE("encode: SmolLM2 tokenizer non-empty output", "[tokenizer][integration]") {
    const fs::path tok_path = PROJECT_ROOT / "models/SmolLM2-135M/tokenizer.json";
    if (!fs::exists(tok_path))
        SKIP("Run download_model.py first");

    Tokenizer tok = load_tokenizer(tok_path.string());
    auto ids = encode(tok, "Hello, world!");
    REQUIRE(!ids.empty());
    for (int32_t id : ids)
        REQUIRE(id >= 0);
    for (int32_t id : ids)
        REQUIRE(id < tok.vocab_size());
}

// ── decode tests ──────────────────────────────────────────────────────────────
TEST_CASE("decode: empty ids → empty string", "[tokenizer][decode]") {
    Tokenizer tok = parse_tokenizer(kEncodeJson);
    std::vector<int32_t> ids;
    REQUIRE(decode(tok, ids).empty());
}

TEST_CASE("decode: single token id", "[tokenizer][decode]") {
    // id 5 = "hi" (BPE chars h+i) → raw bytes 'h','i' → "hi"
    Tokenizer tok = parse_tokenizer(kEncodeJson);
    std::vector<int32_t> ids = {5};
    REQUIRE(decode(tok, ids) == "hi");
}

TEST_CASE("decode: space-prefixed token", "[tokenizer][decode]") {
    // id 14 = "Ġthere" → Ġ=space then 't','h','e','r','e' → " there"
    Tokenizer tok = parse_tokenizer(kEncodeJson);
    std::vector<int32_t> ids = {14};
    REQUIRE(decode(tok, ids) == " there");
}

TEST_CASE("decode: two tokens round-trips encode", "[tokenizer][decode]") {
    Tokenizer tok = parse_tokenizer(kEncodeJson);
    std::string text = "hi there";
    auto ids = encode(tok, text);
    REQUIRE(decode(tok, ids) == text);
}
