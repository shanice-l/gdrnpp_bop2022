#include <torch/extension.h>
#include <vector>

// CUDA forward declarations
// gpu flow

std::vector<torch::Tensor> flow_forward_cuda(
    torch::Tensor depth_src,
    torch::Tensor depth_tgt,
    torch::Tensor KT,
    torch::Tensor Kinv);

// C++ interface

#define CHECK_CUDA(x)                                          \
  do {                                                         \
    TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor"); \
  } while (0)

#define CHECK_CONTIGUOUS(x)                                         \
  do {                                                              \
    TORCH_CHECK(x.is_contiguous(), #x " must be a contiguous tensor"); \
  } while (0)

#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)


std::vector<torch::Tensor> flow_forward (
    torch::Tensor depth_src, // Bx1xHxW
    torch::Tensor depth_tgt,  // Bx1xHxW
    torch::Tensor KT,  // Bx3x4
    torch::Tensor Kinv) {
    // Kinv Bx3x3
    CHECK_INPUT(depth_src);
    CHECK_INPUT(depth_tgt);
    CHECK_INPUT(KT);
    CHECK_INPUT(Kinv);

    return flow_forward_cuda(depth_src, depth_tgt, KT, Kinv);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &flow_forward,
        "flow forward (CUDA)");
}
