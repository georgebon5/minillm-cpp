#pragma once
#include <string_view>

namespace minillm {

struct ModelConfig {
    // ── Dimensions ────────────────────────────────────────────────────────────
    int vocab_size              = 0;
    int hidden_size             = 0;   // d_model: embedding and hidden-state dimension
    int intermediate_size       = 0;   // MLP gate/up projection size
    int num_hidden_layers       = 0;
    int num_attention_heads     = 0;   // query heads (n_heads)
    int num_key_value_heads     = 0;   // KV heads; < n_heads for GQA
    int max_position_embeddings = 0;

    // ── Hyperparameters ───────────────────────────────────────────────────────
    float rms_norm_eps = 1e-5f;
    float rope_theta   = 10000.f;

    // ── Special tokens ────────────────────────────────────────────────────────
    int bos_token_id = 1;
    int eos_token_id = 2;

    // ── Derived (computed after parsing, validated) ───────────────────────────
    int head_dim  = 0;   // hidden_size / num_attention_heads
    int kv_groups = 0;   // num_attention_heads / num_key_value_heads  (GQA ratio)
};

// Parse config.json text into a ModelConfig.
// Throws std::runtime_error on missing required fields or invalid dimensions.
ModelConfig parse_config(std::string_view json_text);

// Load and parse a config.json file from disk.
ModelConfig load_config(std::string_view path);

} // namespace minillm
