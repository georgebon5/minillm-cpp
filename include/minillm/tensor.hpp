#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

namespace minillm {

enum class DType : uint8_t {
    F32,  // 4 bytes per element
    I8,   // 1 byte per element
    I4,   // 0.5 bytes per element (two nibbles packed per byte)
};

// Bytes needed to store `numel` elements of `dtype`.
inline size_t storage_bytes(size_t numel, DType dtype) {
    switch (dtype) {
        case DType::F32: return numel * sizeof(float);
        case DType::I8:  return numel * sizeof(int8_t);
        case DType::I4:  return (numel + 1) / 2;
    }
    return 0;
}

class Tensor {
public:
    Tensor() = default;

    // Allocates a zeroed buffer with row-major strides for the given shape.
    Tensor(std::vector<size_t> shape, DType dtype);

    // ── Metadata ─────────────────────────────────────────────────────────────
    const std::vector<size_t>& shape()   const { return shape_; }
    const std::vector<size_t>& strides() const { return strides_; }
    DType  dtype()  const { return dtype_; }
    size_t ndim()   const { return shape_.size(); }
    size_t numel()  const;
    size_t nbytes() const { return storage_.size(); }

    // ── Raw data ──────────────────────────────────────────────────────────────
    void*       data()       { return storage_.data(); }
    const void* data() const { return storage_.data(); }

    template<typename T>
    T* data_as() { return reinterpret_cast<T*>(storage_.data()); }

    template<typename T>
    const T* data_as() const { return reinterpret_cast<const T*>(storage_.data()); }

private:
    std::vector<size_t>  shape_;
    std::vector<size_t>  strides_;    // row-major, in elements
    DType                dtype_{DType::F32};
    std::vector<uint8_t> storage_;
};

} // namespace minillm
