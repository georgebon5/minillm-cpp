#include "minillm/safetensors.hpp"
#include <nlohmann/json.hpp>

namespace minillm {

static DType dtype_from_string(const std::string& s) {
    if (s == "F32")  return DType::F32;
    if (s == "BF16") return DType::BF16;
    if (s == "I8")   return DType::I8;
    throw std::runtime_error("unsupported safetensors dtype: " + s);
}

std::unordered_map<std::string, TensorInfo>
parse_safetensors_header(const std::string& header_json) {
    using json = nlohmann::json;
    auto root = json::parse(header_json);

    std::unordered_map<std::string, TensorInfo> result;
    for (auto& [name, val] : root.items()) {
        if (name == "__metadata__") continue;

        TensorInfo info;
        info.dtype = dtype_from_string(val.at("dtype").get<std::string>());

        for (auto& d : val.at("shape"))
            info.shape.push_back(d.get<size_t>());

        auto& offsets    = val.at("data_offsets");
        info.offset_start = offsets[0].get<size_t>();
        info.offset_end   = offsets[1].get<size_t>();

        result.emplace(std::move(name), std::move(info));
    }
    return result;
}

} // namespace minillm
