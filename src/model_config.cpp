#include "minillm/model_config.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace minillm {

ModelConfig parse_config(std::string_view json_text) {
    using json = nlohmann::json;
    auto j = json::parse(json_text);

    ModelConfig c;
    c.vocab_size              = j.at("vocab_size").get<int>();
    c.hidden_size             = j.at("hidden_size").get<int>();
    c.intermediate_size       = j.at("intermediate_size").get<int>();
    c.num_hidden_layers       = j.at("num_hidden_layers").get<int>();
    c.num_attention_heads     = j.at("num_attention_heads").get<int>();
    c.num_key_value_heads     = j.at("num_key_value_heads").get<int>();
    c.max_position_embeddings = j.at("max_position_embeddings").get<int>();
    c.rms_norm_eps            = j.at("rms_norm_eps").get<float>();
    c.rope_theta              = j.value("rope_theta", 10000.f);
    c.bos_token_id            = j.at("bos_token_id").get<int>();
    c.eos_token_id            = j.at("eos_token_id").get<int>();

    if (c.hidden_size % c.num_attention_heads != 0)
        throw std::runtime_error(
            "hidden_size must be divisible by num_attention_heads");
    if (c.num_attention_heads % c.num_key_value_heads != 0)
        throw std::runtime_error(
            "num_attention_heads must be divisible by num_key_value_heads");

    c.head_dim  = c.hidden_size / c.num_attention_heads;
    c.kv_groups = c.num_attention_heads / c.num_key_value_heads;

    return c;
}

ModelConfig load_config(std::string_view path) {
    std::ifstream f{std::string(path)};
    if (!f)
        throw std::runtime_error("cannot open config: " + std::string(path));
    std::ostringstream ss;
    ss << f.rdbuf();
    return parse_config(ss.str());
}

} // namespace minillm
