#include "minillm/safetensors.hpp"
#include <nlohmann/json.hpp>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <string>

namespace minillm {

// ── Header parser ─────────────────────────────────────────────────────────────

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

        auto& offsets     = val.at("data_offsets");
        info.offset_start = offsets[0].get<size_t>();
        info.offset_end   = offsets[1].get<size_t>();

        result.emplace(std::move(name), std::move(info));
    }
    return result;
}

// ── SafetensorsFile ───────────────────────────────────────────────────────────

// Reads an 8-byte little-endian uint64 from `p`.
static uint64_t read_le64(const uint8_t* p) {
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i)
        v |= uint64_t(p[i]) << (8 * i);
    return v;
}

SafetensorsFile::SafetensorsFile(std::string_view path) {
    std::string path_str(path);

    fd_ = ::open(path_str.c_str(), O_RDONLY);
    if (fd_ < 0)
        throw std::runtime_error("cannot open: " + path_str + ": " + std::strerror(errno));

    struct stat st{};
    if (::fstat(fd_, &st) < 0) {
        ::close(fd_);
        throw std::runtime_error("fstat failed: " + std::string(std::strerror(errno)));
    }
    mmap_size_ = static_cast<size_t>(st.st_size);

    if (mmap_size_ < 8) {
        ::close(fd_);
        throw std::runtime_error("file too small to be a safetensors file: " + path_str);
    }

    mmap_ptr_ = ::mmap(nullptr, mmap_size_, PROT_READ, MAP_SHARED, fd_, 0);
    if (mmap_ptr_ == MAP_FAILED) {
        ::close(fd_);
        throw std::runtime_error("mmap failed: " + std::string(std::strerror(errno)));
    }

    auto* base = static_cast<const uint8_t*>(mmap_ptr_);

    const uint64_t header_size = read_le64(base);
    if (8 + header_size > mmap_size_) {
        ::munmap(mmap_ptr_, mmap_size_);
        ::close(fd_);
        throw std::runtime_error("header_size exceeds file size");
    }

    std::string header_json(reinterpret_cast<const char*>(base + 8), header_size);
    tensor_map_  = parse_safetensors_header(header_json);
    data_offset_ = 8 + header_size;
}

SafetensorsFile::~SafetensorsFile() {
    if (mmap_ptr_ && mmap_ptr_ != MAP_FAILED)
        ::munmap(mmap_ptr_, mmap_size_);
    if (fd_ >= 0)
        ::close(fd_);
}

Tensor SafetensorsFile::tensor(const std::string& name) const {
    auto it = tensor_map_.find(name);
    if (it == tensor_map_.end())
        throw std::out_of_range("tensor not found: " + name);

    const TensorInfo& info = it->second;
    auto* data_ptr = static_cast<uint8_t*>(mmap_ptr_) + data_offset_ + info.offset_start;
    return Tensor::from_data(data_ptr, info.shape, info.dtype);
}

} // namespace minillm
