#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THC.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>
#include <cfloat>

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include <vector>

// TODO make it in a common file
#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

#define THREADS_PER_BLOCK 512

inline int GET_BLOCKS(const int N) {
  int optimal_block_num = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  int max_block_num = 65000;
  return min(optimal_block_num, max_block_num);
}

template <typename scalar_t>
__global__ void flow_kernel(const int nthreads,
                            const scalar_t *depth_src, const scalar_t *depth_tgt,
                            const int height, const int width,
                            const scalar_t* KT, const scalar_t* Kinv,
                            scalar_t* flow, scalar_t* valid) {
  // batch_size * height * width;
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int w = index % width;
    int h = (index / width) % height;
    int batch_idx = index / width / height;
    scalar_t d_src = depth_src[index];
    Kinv += batch_idx * 9;
    scalar_t x = (w*Kinv[0] + h*Kinv[1] + Kinv[2])*d_src;
    scalar_t y = (w*Kinv[3] + h*Kinv[4] + Kinv[5])*d_src;
    scalar_t z = d_src;
    if (d_src > 1E-3){
        KT += batch_idx * 12;
        scalar_t x_proj = x*KT[0] + y*KT[1] + z*KT[2] + KT[3];
        scalar_t y_proj = x*KT[4] + y*KT[5] + z*KT[6] + KT[7];
        scalar_t z_proj = x*KT[8] + y*KT[9] + z*KT[10] + KT[11] + 1E-15;
        scalar_t w_proj = x_proj / z_proj;
        scalar_t h_proj = y_proj / z_proj;
        int w_proj_i = round(w_proj);
        int h_proj_i = round(h_proj);
        if (w_proj>=0 && w_proj<=width-1 && h_proj>=0 && h_proj<=height-1){
            scalar_t d_tgt = depth_tgt[(batch_idx*height+h_proj_i)*width+w_proj_i];
            if (abs(z_proj - d_tgt) < 3E-3) {
                flow[((batch_idx*2+0)*height+h)*width+w] = h_proj-h;
                flow[((batch_idx*2+1)*height+h)*width+w] = w_proj-w;
                valid[index] = 1;
                return;
            }
        }
    }
    flow[((batch_idx*2+0)*height+h)*width+w] = 0;
    flow[((batch_idx*2+1)*height+h)*width+w] = 0;
    valid[index] = 0;
  } // CUDA_1D_KERNEL_LOOP
} // flow_kernel

std::vector<torch::Tensor> flow_forward_cuda(torch::Tensor depth_src,
                                            torch::Tensor depth_tgt,
                                            torch::Tensor KT,
                                            torch::Tensor Kinv) {
  cudaSetDevice(depth_src.get_device());

  const int batch_size = depth_src.size(0);
  const int height = depth_src.size(2);
  const int width = depth_src.size(3);

  auto flow = torch::zeros({batch_size, 2, height, width}, depth_src.options());
  auto valid = torch::zeros({batch_size, 1, height, width}, depth_src.options());

  AT_DISPATCH_FLOATING_TYPES(depth_src.scalar_type(), "flow_forward_cuda", ([&] {
    const int count = batch_size * height * width;
    flow_kernel<scalar_t><<<GET_BLOCKS(count), THREADS_PER_BLOCK>>>(
        count,
        depth_src.data_ptr<scalar_t>(),
        depth_tgt.data_ptr<scalar_t>(),
        height, width,
        KT.data_ptr<scalar_t>(),
        Kinv.data_ptr<scalar_t>(),
        flow.data_ptr<scalar_t>(),
        valid.data_ptr<scalar_t>()
      );
  }));
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in flow_forward_cuda: %s\n", cudaGetErrorString(err));
  }
  return {flow, valid};
}
