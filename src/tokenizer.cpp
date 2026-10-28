#include "minillm/tokenizer.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <stdexcept>

namespace minillm {

Tokenizer parse_tokenizer(std::string_view json_text) {
    auto j = nlohmann::json::parse(json_text);

    Tokenizer tok;

    // ── vocab: id → token string ──────────────────────────────────────────────
    // tokenizer.json stores vocab as {"token": id, ...} inside "model.vocab"
    const auto& vocab = j.at("model").at("vocab");
    tok.id_to_token.resize(vocab.size());
    for (auto& [token_str, id_val] : vocab.items()) {
        int32_t id = id_val.get<int32_t>();
        tok.token_to_id[token_str] = id;
        if (static_cast<size_t>(id) >= tok.id_to_token.size())
            tok.id_to_token.resize(static_cast<size_t>(id) + 1);
        tok.id_to_token[static_cast<size_t>(id)] = token_str;
    }

    // ── BPE merges ────────────────────────────────────────────────────────────
    // stored as array of ["left", "right"] pairs; index = priority
    const auto& merges = j.at("model").at("merges");
    tok.merge_ranks.reserve(merges.size());
    for (size_t i = 0; i < merges.size(); ++i) {
        // Each merge is either a string "left right" or a ["left","right"] array
        std::string key;
        if (merges[i].is_string()) {
            key = merges[i].get<std::string>();
        } else {
            key = merges[i][0].get<std::string>() + " " + merges[i][1].get<std::string>();
        }
        tok.merge_ranks[key] = static_cast<int32_t>(i);
    }

    // ── added_tokens (special tokens) ─────────────────────────────────────────
    if (j.contains("added_tokens")) {
        for (const auto& entry : j.at("added_tokens")) {
            int32_t id         = entry.at("id").get<int32_t>();
            std::string content = entry.at("content").get<std::string>();
            // Insert into vocab maps if not already there
            tok.token_to_id[content] = id;
            if (static_cast<size_t>(id) >= tok.id_to_token.size())
                tok.id_to_token.resize(static_cast<size_t>(id) + 1);
            tok.id_to_token[static_cast<size_t>(id)] = content;

            // Detect special token roles by "special": true + content heuristics
            if (!entry.contains("special") || !entry.at("special").get<bool>())
                continue;
            const std::string& c = content;
            if (c == "<unk>")         tok.unk_id = id;
            else if (c == "<s>")      tok.bos_id = id;
            else if (c == "</s>")     tok.eos_id = id;
        }
    }

    return tok;
}

Tokenizer load_tokenizer(std::string_view path) {
    std::ifstream f{std::string(path)};
    if (!f)
        throw std::runtime_error("Cannot open tokenizer file: " + std::string(path));
    std::string text{std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
    return parse_tokenizer(text);
}

} // namespace minillm
