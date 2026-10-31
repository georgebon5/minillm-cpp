#include "minillm/safetensors.hpp"
#include <nlohmann/json.hpp>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <algorithm>
#include <cassert>
#include <cerrno>
#include <cmath>
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

// ── BF16 → F32 conversion ─────────────────────────────────────────────────────

// BF16 is the top 16 bits of F32 (same exponent, fewer mantissa bits).
// Conversion: zero-extend to 32 bits and reinterpret.
// memcpy is required here — casting via pointer would be UB (strict aliasing).
static float bf16_to_f32(uint16_t bf16) {
    uint32_t bits = uint32_t(bf16) << 16;
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

Tensor SafetensorsFile::tensor_f32(const std::string& name) const {
    Tensor raw = tensor(name);

    if (raw.dtype() == DType::F32) {
        Tensor out(raw.shape(), DType::F32);
        std::memcpy(out.data(), raw.data(), raw.nbytes());
        return out;
    }

    if (raw.dtype() == DType::BF16) {
        Tensor out(raw.shape(), DType::F32);
        const auto* src = raw.data_as<uint16_t>();
              auto* dst = out.data_as<float>();
        const size_t n  = raw.numel();
        for (size_t i = 0; i < n; ++i)
            dst[i] = bf16_to_f32(src[i]);
        return out;
    }

    throw std::runtime_error("tensor_f32: unsupported dtype for tensor '" + name + "'");
}

QuantTensor SafetensorsFile::tensor_q(const std::string& name, DType qdtype) const {
    Tensor raw = tensor(name);
    assert(raw.ndim() == 2);
    const size_t rows = raw.shape()[0];
    const size_t cols = raw.shape()[1];
    const float  qmax = (qdtype == DType::I8) ? 127.f : 7.f;

    QuantTensor qt;
    qt.data = Tensor({rows, cols}, qdtype);
    qt.scales.resize(rows, 1.f);

    std::vector<float> row(cols);
    const auto* src_bf16 = (raw.dtype() == DType::BF16) ? raw.data_as<uint16_t>() : nullptr;
    const auto* src_f32  = (raw.dtype() == DType::F32)  ? raw.data_as<float>()    : nullptr;

    for (size_t r = 0; r < rows; ++r) {
        // Convert one row to F32 (O(cols) temporary, never the full matrix).
        if (src_bf16) {
            for (size_t c = 0; c < cols; ++c)
                row[c] = bf16_to_f32(src_bf16[r * cols + c]);
        } else {
            std::memcpy(row.data(), src_f32 + r * cols, cols * sizeof(float));
        }

        float max_abs = 0.f;
        for (float v : row) max_abs = std::max(max_abs, std::abs(v));
        if (max_abs == 0.f) continue;
        qt.scales[r] = max_abs / qmax;

        if (qdtype == DType::I8) {
            auto* dst = reinterpret_cast<int8_t*>(qt.data.data()) + r * cols;
            for (size_t c = 0; c < cols; ++c)
                dst[c] = static_cast<int8_t>(std::round(row[c] / qt.scales[r]));
        } else {
            // INT4 nibble-packed: two values per byte
            auto* dst = reinterpret_cast<uint8_t*>(qt.data.data()) + r * (cols / 2);
            for (size_t c = 0; c < cols; c += 2) {
                int lo = static_cast<int>(std::round(row[c]     / qt.scales[r]));
                int hi = static_cast<int>(std::round(row[c + 1] / qt.scales[r]));
                lo = std::max(-7, std::min(7, lo));
                hi = std::max(-7, std::min(7, hi));
                dst[c / 2] = static_cast<uint8_t>((lo & 0xF) | ((hi & 0xF) << 4));
            }
        }
    }
    return qt;
}

std::unordered_map<std::string, Tensor> SafetensorsFile::load_all_f32() const {
    std::unordered_map<std::string, Tensor> result;
    result.reserve(tensor_map_.size());
    for (const auto& [name, _] : tensor_map_)
        result.emplace(name, tensor_f32(name));
    return result;
}

} // namespace minillm
