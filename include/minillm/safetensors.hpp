#pragma once
#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>
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

// ── File loader ───────────────────────────────────────────────────────────────

// Opens a .safetensors file and memory-maps it.
// Tensor views (non-owning) are valid only while this object is alive.
class SafetensorsFile {
public:
    explicit SafetensorsFile(std::string_view path);
    ~SafetensorsFile();

    // Non-copyable, non-movable for simplicity.
    SafetensorsFile(const SafetensorsFile&) = delete;
    SafetensorsFile& operator=(const SafetensorsFile&) = delete;

    // Returns a non-owning Tensor view into the mmap'd data.
    // Throws std::out_of_range if `name` is not found.
    Tensor tensor(const std::string& name) const;

    bool has(const std::string& name) const { return tensor_map_.count(name) > 0; }

    const std::unordered_map<std::string, TensorInfo>& tensor_map() const {
        return tensor_map_;
    }

private:
    int    fd_{-1};
    void*  mmap_ptr_{nullptr};
    size_t mmap_size_{0};
    size_t data_offset_{0};   // byte offset of the tensor data region within the file
    std::unordered_map<std::string, TensorInfo> tensor_map_;
};

} // namespace minillm
