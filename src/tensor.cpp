#include "minillm/tensor.hpp"

namespace minillm {

static std::vector<size_t> make_strides(const std::vector<size_t>& shape) {
    std::vector<size_t> st(shape.size(), 1);
    for (int i = (int)shape.size() - 2; i >= 0; --i)
        st[i] = st[i + 1] * shape[i + 1];
    return st;
}

Tensor::Tensor(std::vector<size_t> shape, DType dtype)
    : shape_(std::move(shape))
    , strides_(make_strides(shape_))
    , dtype_(dtype)
    , storage_(storage_bytes(numel(), dtype), 0)
    , ptr_(storage_.data())
{}

// Private constructor for non-owning views.
Tensor::Tensor(void* data, std::vector<size_t> shape, DType dtype)
    : shape_(std::move(shape))
    , strides_(make_strides(shape_))
    , dtype_(dtype)
    , storage_()                               // stays empty — we don't own this memory
    , ptr_(static_cast<uint8_t*>(data))
{}

Tensor Tensor::from_data(void* data, std::vector<size_t> shape, DType dtype) {
    return Tensor(data, std::move(shape), dtype);
}

size_t Tensor::numel() const {
    if (shape_.empty()) return 0;
    size_t n = 1;
    for (size_t d : shape_) n *= d;
    return n;
}

size_t Tensor::flat_offset(std::initializer_list<size_t> idx) const {
    assert(idx.size() == ndim() && "index rank must match tensor rank");
    size_t offset = 0, d = 0;
    for (size_t i : idx) {
        assert(i < shape_[d] && "index out of bounds");
        offset += i * strides_[d++];
    }
    return offset;
}

} // namespace minillm
