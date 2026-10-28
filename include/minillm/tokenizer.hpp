#pragma once
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace minillm {

struct Tokenizer {
    // token string → integer id
    std::unordered_map<std::string, int32_t> token_to_id;

    // integer id → token string  (indexed by id, so id_to_token[i] = token with id i)
    std::vector<std::string> id_to_token;

    // BPE merge rules: key = "left right" (the two subwords separated by a space),
    // value = priority (0 = first rule, i.e. highest priority).
    std::unordered_map<std::string, int32_t> merge_ranks;

    // Special-token ids (set from added_tokens or defaults).
    int32_t bos_id =  1;
    int32_t eos_id =  2;
    int32_t unk_id =  0;

    int32_t vocab_size() const { return static_cast<int32_t>(id_to_token.size()); }
};

// Parse tokenizer.json content from a string.
Tokenizer parse_tokenizer(std::string_view json_text);

// Load and parse a tokenizer.json file from disk.
Tokenizer load_tokenizer(std::string_view path);

// Encode plain text → token ids (no BOS/EOS added).
std::vector<int32_t> encode(const Tokenizer& tok, std::string_view text);

// Decode token ids → plain text (skips unknown ids).
std::string decode(const Tokenizer& tok, std::span<const int32_t> ids);

} // namespace minillm
