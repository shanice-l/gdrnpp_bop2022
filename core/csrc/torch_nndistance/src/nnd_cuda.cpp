#include <torch/torch.h>
#include <vector>


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


int NmDistanceKernelLauncher(
    at::Tensor xyz1,
    at::Tensor xyz2,
    at::Tensor dist1,
    at::Tensor dist2,
    at::Tensor idx1,
    at::Tensor idx2);
int NmDistanceGradKernelLauncher(
    at::Tensor xyz1,
    at::Tensor xyz2,
    at::Tensor gradxyz1,
    at::Tensor gradxyz2,
    at::Tensor graddist1,
    at::Tensor graddist2,
    at::Tensor idx1,
    at::Tensor idx2);

int nnd_forward_cuda(
    at::Tensor xyz1,
    at::Tensor xyz2,
    at::Tensor dist1,
    at::Tensor dist2,
    at::Tensor idx1,
    at::Tensor idx2) {
      CHECK_INPUT(xyz1);
      CHECK_INPUT(xyz2);
      CHECK_INPUT(dist1);
      CHECK_INPUT(dist2);
      CHECK_INPUT(idx1);
      CHECK_INPUT(idx2);


    return NmDistanceKernelLauncher(xyz1, xyz2, dist1, dist2, idx1, idx2);
}


int nnd_backward_cuda(
    at::Tensor xyz1,
    at::Tensor xyz2,
    at::Tensor gradxyz1,
    at::Tensor gradxyz2,
    at::Tensor graddist1,
    at::Tensor graddist2,
    at::Tensor idx1,
    at::Tensor idx2)
{
    CHECK_INPUT(xyz1);
    CHECK_INPUT(xyz2);
    CHECK_INPUT(gradxyz1);
    CHECK_INPUT(gradxyz2);
    CHECK_INPUT(graddist1);
    CHECK_INPUT(graddist2);
    CHECK_INPUT(idx1);
    CHECK_INPUT(idx2);

    return NmDistanceGradKernelLauncher(
        xyz1,
        xyz2,
        gradxyz1,
        gradxyz2,
        graddist1,
        graddist2,
        idx1,
        idx2);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("nnd_forward_cuda", &nnd_forward_cuda, "NND forward (CUDA)");
  m.def("nnd_backward_cuda", &nnd_backward_cuda, "NND backward (CUDA)");
}
