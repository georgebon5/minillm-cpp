#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>
#include "minillm/tokenizer.hpp"
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace minillm;
namespace fs = std::filesystem;

static const fs::path PROJECT_ROOT  = MINILLM_PROJECT_ROOT;
static const fs::path TOK_PATH      = PROJECT_ROOT / "models/SmolLM2-135M/tokenizer.json";
static const fs::path REF_PATH      = PROJECT_ROOT / "tools/reference_tokenizer.json";

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
