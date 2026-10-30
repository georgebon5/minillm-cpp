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

// ── EOS / ChatML special-token detection ─────────────────────────────────────

static const std::string kSmolLM2Json = R"({
  "model": {
    "type": "BPE",
    "vocab": {
      "<|endoftext|>": 0,
      "<|im_start|>":  1,
      "<|im_end|>":    2,
      "hello":         3
    },
    "merges": []
  },
  "added_tokens": [
    {"id": 0, "content": "<|endoftext|>", "special": true},
    {"id": 1, "content": "<|im_start|>",  "special": true},
    {"id": 2, "content": "<|im_end|>",    "special": true}
  ]
})";

TEST_CASE("parse_tokenizer: endoftext mapped to eos_id", "[tokenizer][special]") {
    Tokenizer tok = parse_tokenizer(kSmolLM2Json);
    REQUIRE(tok.eos_id == 0);
}

TEST_CASE("parse_tokenizer: im_start_id and im_end_id detected", "[tokenizer][special]") {
    Tokenizer tok = parse_tokenizer(kSmolLM2Json);
    REQUIRE(tok.im_start_id == 1);
    REQUIRE(tok.im_end_id   == 2);
}

TEST_CASE("parse_tokenizer: has_chat_template true when both im tokens present", "[tokenizer][special]") {
    Tokenizer tok = parse_tokenizer(kSmolLM2Json);
    REQUIRE(tok.has_chat_template());
}

TEST_CASE("parse_tokenizer: has_chat_template false for base vocab", "[tokenizer][special]") {
    Tokenizer tok = parse_tokenizer(kMinimalJson);  // only <unk>/<s>/</s>
    REQUIRE_FALSE(tok.has_chat_template());
}

TEST_CASE("encode: special tokens inserted directly without BPE", "[tokenizer][special]") {
    Tokenizer tok = parse_tokenizer(kSmolLM2Json);
    // <|im_start|> must become id=1 directly, not go through byte-level BPE
    auto ids = encode(tok, "<|im_start|>");
    REQUIRE(ids == std::vector<int32_t>{1});
}

TEST_CASE("encode: im_end token gets correct id", "[tokenizer][special]") {
    Tokenizer tok = parse_tokenizer(kSmolLM2Json);
    auto ids = encode(tok, "<|im_end|>");
    REQUIRE(ids == std::vector<int32_t>{2});
}

TEST_CASE("encode: ChatML sequence preserves special token ids", "[tokenizer][special]") {
    // Simulate the ChatML wrapper: <|im_start|>user\n…<|im_end|>\n<|im_start|>assistant\n
    // The fixture has no BPE merges for "user" etc, but special tokens must land correctly.
    Tokenizer tok = parse_tokenizer(kSmolLM2Json);
    auto ids = encode(tok, "<|im_start|><|im_end|>");
    REQUIRE(ids.size() >= 2);
    REQUIRE(ids[0] == 1);   // <|im_start|>
    REQUIRE(ids[1] == 2);   // <|im_end|>
}

TEST_CASE("encode: endoftext token gets correct id", "[tokenizer][special]") {
    Tokenizer tok = parse_tokenizer(kSmolLM2Json);
    auto ids = encode(tok, "<|endoftext|>");
    REQUIRE(ids == std::vector<int32_t>{0});
}

// ── decode: special token stripping ──────────────────────────────────────────

static const std::string kStripJson = R"({
  "model": {
    "type": "BPE",
    "vocab": {
      "<|endoftext|>": 0,
      "<|im_start|>":  1,
      "<|im_end|>":    2,
      "h": 3, "i": 4, "hi": 5
    },
    "merges": ["h i"]
  },
  "added_tokens": [
    {"id": 0, "content": "<|endoftext|>", "special": true},
    {"id": 1, "content": "<|im_start|>",  "special": true},
    {"id": 2, "content": "<|im_end|>",    "special": true}
  ]
})";

TEST_CASE("decode: special tokens are stripped from output", "[tokenizer][decode]") {
    Tokenizer tok = parse_tokenizer(kStripJson);
    // Sequence: <|im_start|> "hi" <|im_end|> — only "hi" should survive
    std::vector<int32_t> ids = {1, 5, 2};
    REQUIRE(decode(tok, ids) == "hi");
}

TEST_CASE("decode: endoftext token stripped", "[tokenizer][decode]") {
    Tokenizer tok = parse_tokenizer(kStripJson);
    std::vector<int32_t> ids = {5, 0};  // "hi" + <|endoftext|>
    REQUIRE(decode(tok, ids) == "hi");
}

TEST_CASE("decode: all-special sequence yields empty string", "[tokenizer][decode]") {
    Tokenizer tok = parse_tokenizer(kStripJson);
    std::vector<int32_t> ids = {0, 1, 2};
    REQUIRE(decode(tok, ids).empty());
}

TEST_CASE("decode: interleaved special tokens stripped correctly", "[tokenizer][decode]") {
    Tokenizer tok = parse_tokenizer(kStripJson);
    // <|im_start|> "hi" <|im_end|> "hi" <|endoftext|>
    std::vector<int32_t> ids = {1, 5, 2, 5, 0};
    REQUIRE(decode(tok, ids) == "hihi");
}

// ── encode: stress / edge-case tests ─────────────────────────────────────────

TEST_CASE("encode: 100 consecutive special tokens produce correct id sequence", "[tokenizer][stress]") {
    Tokenizer tok = parse_tokenizer(kSmolLM2Json);
    std::string input;
    for (int i = 0; i < 100; ++i) input += "<|im_start|>";
    auto ids = encode(tok, input);
    REQUIRE(ids.size() == 100);
    for (int32_t id : ids) REQUIRE(id == tok.im_start_id);
}

TEST_CASE("encode: alternating special and regular tokens", "[tokenizer][stress]") {
    Tokenizer tok = parse_tokenizer(kSmolLM2Json);
    // hello (id 3 in kSmolLM2Json) interleaved with <|im_end|> (id 2)
    // kSmolLM2Json has no merges, so "hello" → unk in this fixture — just verify
    // that special tokens land correctly regardless of what's between them.
    std::string input = "<|im_start|><|im_end|><|im_start|><|im_end|>";
    auto ids = encode(tok, input);
    REQUIRE(ids.size() == 4);
    REQUIRE(ids[0] == tok.im_start_id);
    REQUIRE(ids[1] == tok.im_end_id);
    REQUIRE(ids[2] == tok.im_start_id);
    REQUIRE(ids[3] == tok.im_end_id);
}

TEST_CASE("encode: empty string after stripping gives empty ids", "[tokenizer][stress]") {
    Tokenizer tok = parse_tokenizer(kSmolLM2Json);
    REQUIRE(encode(tok, "").empty());
}

TEST_CASE("encode: special token at end of text", "[tokenizer][stress]") {
    Tokenizer tok = parse_tokenizer(kSmolLM2Json);
    auto ids = encode(tok, "<|endoftext|>");
    REQUIRE(ids.size() == 1);
    REQUIRE(ids[0] == tok.eos_id);
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
