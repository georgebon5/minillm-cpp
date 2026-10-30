#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>
#include "minillm/tokenizer.hpp"
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace minillm;
namespace fs = std::filesystem;

static const fs::path PROJECT_ROOT      = MINILLM_PROJECT_ROOT;
static const fs::path TOK_PATH          = PROJECT_ROOT / "models/SmolLM2-135M/tokenizer.json";
static const fs::path TOK_INSTRUCT_PATH = PROJECT_ROOT / "models/SmolLM2-360M-Instruct/tokenizer.json";
static const fs::path REF_PATH          = PROJECT_ROOT / "tools/reference_tokenizer.json";

TEST_CASE("tokenizer round-trip: encode matches HuggingFace", "[tokenizer][integration]") {
    if (!fs::exists(TOK_PATH) || !fs::exists(REF_PATH))
        SKIP("Run download_model.py and gen_tokenizer_refs.py first");

    Tokenizer tok = load_tokenizer(TOK_PATH.string());

    std::ifstream rf(REF_PATH);
    auto ref = nlohmann::json::parse(rf);

    for (const auto& sample : ref.at("samples")) {
        const std::string text      = sample.at("text").get<std::string>();
        const auto expected         = sample.at("ids").get<std::vector<int32_t>>();
        const auto got              = encode(tok, text);

        INFO("text: \"" << text << "\"");
        REQUIRE(got == expected);
    }
}

// ── Instruct / ChatML integration tests ──────────────────────────────────────

TEST_CASE("instruct tokenizer: has_chat_template is true", "[tokenizer][integration][instruct]") {
    if (!fs::exists(TOK_INSTRUCT_PATH))
        SKIP("Run: python tools/download_model.py --size 360M --instruct");

    Tokenizer tok = load_tokenizer(TOK_INSTRUCT_PATH.string());
    REQUIRE(tok.has_chat_template());
    REQUIRE(tok.im_start_id >= 0);
    REQUIRE(tok.im_end_id   >= 0);
}

TEST_CASE("instruct tokenizer: im_start encodes to single id", "[tokenizer][integration][instruct]") {
    if (!fs::exists(TOK_INSTRUCT_PATH))
        SKIP("Run: python tools/download_model.py --size 360M --instruct");

    Tokenizer tok = load_tokenizer(TOK_INSTRUCT_PATH.string());
    auto ids = encode(tok, "<|im_start|>");
    REQUIRE(ids.size() == 1);
    REQUIRE(ids[0] == tok.im_start_id);
}

TEST_CASE("instruct tokenizer: im_end encodes to single id", "[tokenizer][integration][instruct]") {
    if (!fs::exists(TOK_INSTRUCT_PATH))
        SKIP("Run: python tools/download_model.py --size 360M --instruct");

    Tokenizer tok = load_tokenizer(TOK_INSTRUCT_PATH.string());
    auto ids = encode(tok, "<|im_end|>");
    REQUIRE(ids.size() == 1);
    REQUIRE(ids[0] == tok.im_end_id);
}

TEST_CASE("instruct tokenizer: ChatML prefix has correct first two token ids", "[tokenizer][integration][instruct]") {
    if (!fs::exists(TOK_INSTRUCT_PATH))
        SKIP("Run: python tools/download_model.py --size 360M --instruct");

    Tokenizer tok = load_tokenizer(TOK_INSTRUCT_PATH.string());
    // "<|im_start|>user\n" must start with im_start_id
    const std::string chatml = "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n";
    auto ids = encode(tok, chatml);

    REQUIRE(!ids.empty());
    REQUIRE(ids.front() == tok.im_start_id);
    // Must also contain im_end_id somewhere in the middle
    bool found_end = false;
    for (int32_t id : ids) if (id == tok.im_end_id) { found_end = true; break; }
    REQUIRE(found_end);
}

TEST_CASE("instruct tokenizer: special_tokens list is non-empty and sorted longest-first", "[tokenizer][integration][instruct]") {
    if (!fs::exists(TOK_INSTRUCT_PATH))
        SKIP("Run: python tools/download_model.py --size 360M --instruct");

    Tokenizer tok = load_tokenizer(TOK_INSTRUCT_PATH.string());
    REQUIRE(!tok.special_tokens.empty());
    for (size_t i = 1; i < tok.special_tokens.size(); ++i)
        REQUIRE(tok.special_tokens[i-1].first.size() >= tok.special_tokens[i].first.size());
}

TEST_CASE("tokenizer round-trip: decode(encode(text)) == text", "[tokenizer][integration]") {
    if (!fs::exists(TOK_PATH) || !fs::exists(REF_PATH))
        SKIP("Run download_model.py and gen_tokenizer_refs.py first");

    Tokenizer tok = load_tokenizer(TOK_PATH.string());

    std::ifstream rf(REF_PATH);
    auto ref = nlohmann::json::parse(rf);

    for (const auto& sample : ref.at("samples")) {
        const std::string text = sample.at("text").get<std::string>();
        const auto ids         = encode(tok, text);
        const std::string got  = decode(tok, ids);

        INFO("text: \"" << text << "\"");
        REQUIRE(got == text);
    }
}
