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
{}

size_t Tensor::numel() const {
    if (shape_.empty()) return 0;
    size_t n = 1;
    for (size_t d : shape_) n *= d;
    return n;
}

} // namespace minillm
