#pragma once
#include <cstddef>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include "tensor.hpp"

namespace minillm {

struct TensorInfo {
    DType               dtype;
    std::vector<size_t> shape;
    size_t              offset_start;  // byte offset into the data region
    size_t              offset_end;
};

// Parse the JSON header string extracted from a .safetensors file.
// The "__metadata__" key is silently ignored.
// Throws std::runtime_error on parse errors or unsupported dtypes.
std::unordered_map<std::string, TensorInfo>
parse_safetensors_header(const std::string& header_json);

} // namespace minillm
