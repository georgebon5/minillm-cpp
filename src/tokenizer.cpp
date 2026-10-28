#include "minillm/tokenizer.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <stdexcept>
#include <climits>
#include <limits>
#include <list>

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

// ── BPE encode ────────────────────────────────────────────────────────────────

// GPT-2 byte-level mapping: each raw byte → single unicode codepoint string.
// The mapping avoids control/whitespace characters by offsetting them into
// printable Latin-Extended range (same table as GPT-2's bytes_to_unicode()).
static const std::string& byte_to_bpe_char(uint8_t b) {
    static std::vector<std::string> table = []() {
        std::vector<std::string> t(256);
        // Printable ASCII range that maps to itself: 33-126 + 161-172 + 174-255
        auto encode_cp = [](uint32_t cp) -> std::string {
            std::string s;
            if (cp < 0x80) {
                s += static_cast<char>(cp);
            } else if (cp < 0x800) {
                s += static_cast<char>(0xC0 | (cp >> 6));
                s += static_cast<char>(0x80 | (cp & 0x3F));
            } else {
                s += static_cast<char>(0xE0 | (cp >> 12));
                s += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                s += static_cast<char>(0x80 | (cp & 0x3F));
            }
            return s;
        };

        // First pass: mark which bytes map to themselves
        std::vector<bool> direct(256, false);
        for (int c = '!'; c <= '~'; ++c) direct[c] = true;
        for (int c = 0xA1; c <= 0xAC; ++c) direct[c] = true;
        for (int c = 0xAE; c <= 0xFF; ++c) direct[c] = true;

        // Second pass: assign codepoints; non-direct bytes get codepoints
        // starting at 256, in order.
        uint32_t extra = 256;
        for (int b = 0; b < 256; ++b) {
            if (direct[b])
                t[b] = encode_cp(static_cast<uint32_t>(b));
            else
                t[b] = encode_cp(extra++);
        }
        return t;
    }();
    return table[b];
}

// Split UTF-8 string into a list of BPE character tokens.
// Each input byte is independently mapped to its BPE char representation,
// then adjacent chars belonging to the same "pre-token word" are grouped.
// For simplicity we do a GPT-2-style pre-tokenization: split on whitespace
// boundaries, prepending Ġ (U+0120, encoded as \xC4\x80) to each non-first word.
static std::vector<std::string> pretokenize(std::string_view text) {
    // Space byte (0x20) maps to U+0120 (Ġ) in GPT-2's byte-level encoding.
    // Derive it from byte_to_bpe_char so it's always consistent with merge keys.
    const std::string kSpace = byte_to_bpe_char(0x20);

    std::vector<std::string> words;
    // Walk the raw bytes; group into "words" separated by space (0x20).
    // A space byte becomes the Ġ prefix of the next word, not its own token.
    bool at_start = true;
    std::string cur;
    for (size_t i = 0; i < text.size(); ++i) {
        uint8_t b = static_cast<uint8_t>(text[i]);
        if (b == ' ') {
            if (!cur.empty()) { words.push_back(cur); cur.clear(); }
            at_start = false;
            // next word will get kSpace prefix
            cur = kSpace;
        } else {
            if (at_start && cur.empty()) {
                // first word: no prefix
            }
            cur += byte_to_bpe_char(b);
        }
    }
    if (!cur.empty() && cur != kSpace) words.push_back(cur);
    return words;
}

// BPE merge loop on a single pre-token word.
// Returns the list of final sub-word tokens for that word.
static std::vector<std::string> bpe_word(
        const std::unordered_map<std::string, int32_t>& merge_ranks,
        const std::string& word_chars)
{
    // Start: each BPE char is its own token.
    // word_chars is already a concatenated string of BPE chars — but we need
    // to split it back into individual codepoints (each BPE char is 1-3 bytes).
    std::list<std::string> tokens;
    {
        size_t i = 0;
        const auto* p = reinterpret_cast<const uint8_t*>(word_chars.data());
        while (i < word_chars.size()) {
            uint8_t c = p[i];
            size_t len = 1;
            if ((c & 0xE0) == 0xC0)      len = 2;
            else if ((c & 0xF0) == 0xE0) len = 3;
            else if ((c & 0xF8) == 0xF0) len = 4;
            tokens.push_back(word_chars.substr(i, len));
            i += len;
        }
    }

    // Repeatedly find lowest-rank adjacent pair and merge.
    while (tokens.size() > 1) {
        int32_t best_rank = std::numeric_limits<int32_t>::max();
        auto best_it = tokens.end();

        for (auto it = tokens.begin(); it != tokens.end(); ++it) {
            auto next = std::next(it);
            if (next == tokens.end()) break;
            std::string key = *it + " " + *next;
            auto found = merge_ranks.find(key);
            if (found != merge_ranks.end() && found->second < best_rank) {
                best_rank = found->second;
                best_it = it;
            }
        }

        if (best_it == tokens.end()) break; // no merge possible

        // Merge best_it and its successor in-place.
        auto next = std::next(best_it);
        *best_it += *next;
        tokens.erase(next);
    }

    return {tokens.begin(), tokens.end()};
}

std::vector<int32_t> encode(const Tokenizer& tok, std::string_view text) {
    std::vector<int32_t> ids;
    for (const std::string& word : pretokenize(text)) {
        for (const std::string& token : bpe_word(tok.merge_ranks, word)) {
            auto it = tok.token_to_id.find(token);
            if (it != tok.token_to_id.end())
                ids.push_back(it->second);
            else
                ids.push_back(tok.unk_id);
        }
    }
    return ids;
}

} // namespace minillm
