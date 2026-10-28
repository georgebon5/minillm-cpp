#pragma once
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
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

    // Owning: allocates a zeroed buffer with row-major strides.
    Tensor(std::vector<size_t> shape, DType dtype);

    // Non-owning view into existing memory (e.g. an mmap'd safetensors buffer).
    // The caller must ensure the memory outlives this Tensor.
    static Tensor from_data(void* data, std::vector<size_t> shape, DType dtype);

    // ── Metadata ──────────────────────────────────────────────────────────────
    const std::vector<size_t>& shape()   const { return shape_; }
    const std::vector<size_t>& strides() const { return strides_; }
    DType  dtype()    const { return dtype_; }
    size_t ndim()     const { return shape_.size(); }
    size_t numel()    const;
    size_t nbytes()   const { return storage_bytes(numel(), dtype_); }
    bool   owns_data() const { return !storage_.empty(); }

    // ── Raw data ──────────────────────────────────────────────────────────────
    void*       data()       { return ptr_; }
    const void* data() const { return ptr_; }

    template<typename T>
    T* data_as() { return reinterpret_cast<T*>(ptr_); }

    template<typename T>
    const T* data_as() const { return reinterpret_cast<const T*>(ptr_); }

    // ── Element access ────────────────────────────────────────────────────────
    // Computes flat offset via strides: offset = sum(idx[d] * strides[d]).
    // Usage: t.at<float>({row, col})
    template<typename T>
    T& at(std::initializer_list<size_t> idx) {
        return data_as<T>()[flat_offset(idx)];
    }

    template<typename T>
    const T& at(std::initializer_list<size_t> idx) const {
        return data_as<T>()[flat_offset(idx)];
    }

private:
    // Private constructor used by from_data().
    Tensor(void* data, std::vector<size_t> shape, DType dtype);

    size_t flat_offset(std::initializer_list<size_t> idx) const;

    std::vector<size_t>  shape_;
    std::vector<size_t>  strides_;     // row-major, in elements
    DType                dtype_{DType::F32};
    std::vector<uint8_t> storage_;     // non-empty only for owning tensors
    uint8_t*             ptr_{nullptr}; // always valid; points into storage_ or external mem
};

} // namespace minillm
