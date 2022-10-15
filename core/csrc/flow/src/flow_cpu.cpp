#include <torch/extension.h>
#include <vector>

template <typename scalar_t>
void flow_kernel(const int nthreads,
                 const scalar_t *depth_src, const scalar_t *depth_tgt,
                 int height, int width,
                 const scalar_t* KT, const scalar_t* Kinv,
                 scalar_t* flow, scalar_t* valid) {
  // batch_size * height * width;
  for (int index = 0; index < nthreads; index++) {
    int w = index % width;
    int h = (index / width) % height;
    int batch_idx = index / width / height;
    scalar_t d_src = depth_src[index];
    // scalar_t d_tgt = depth_tgt[index];
    Kinv += batch_idx * 9;
    scalar_t x = (w*Kinv[0] + h*Kinv[1] + Kinv[2])*d_src;
    scalar_t y = (w*Kinv[3] + h*Kinv[4] + Kinv[5])*d_src;
    scalar_t z = d_src;
    int _point_valid = 0;
    if (d_src > 1E-3) {
      KT += batch_idx * 12;
      scalar_t x_proj = x*KT[0] + y*KT[1] + z*KT[2] + KT[3];
      scalar_t y_proj = x*KT[4] + y*KT[5] + z*KT[6] + KT[7];
      scalar_t z_proj = x*KT[8] + y*KT[9] + z*KT[10] + KT[11] + 1E-15;
      scalar_t w_proj = x_proj / z_proj;
      scalar_t h_proj = y_proj / z_proj;
      int w_proj_i = round(w_proj);
      int h_proj_i = round(h_proj);
      if (w_proj>=0 && w_proj<=width-1 && h_proj>=0 && h_proj<=height-1) {
        scalar_t d_tgt = depth_tgt[(batch_idx*height+h_proj_i)*width+w_proj_i];
        if (abs(z_proj - d_tgt) < 3E-3) {
          flow[((batch_idx*2+0)*height+h)*width+w] = h_proj-h;
          flow[((batch_idx*2+1)*height+h)*width+w] = w_proj-w;
          valid[index] = 1;
          _point_valid = 1;
        }
      }
    } //if (d_src > 1E-3)
    if (_point_valid == 0) {
      flow[((batch_idx*2+0)*height+h)*width+w] = 0;
      flow[((batch_idx*2+1)*height+h)*width+w] = 0;
      valid[index] = 0;
    }
  } //for (int index = 0; index < nthreads; index++)
}

std::vector<torch::Tensor> flow_cpu_forward(
    torch::Tensor depth_src,  // Bx1xHxW
    torch::Tensor depth_tgt, //Bx1xHxW
    torch::Tensor KT,  //Bx3x4
    torch::Tensor Kinv) {
    // Kinv Bx3x3
  const int batch_size = depth_src.size(0);
  const int height = depth_src.size(2);
  const int width = depth_src.size(3);
  auto flow = torch::zeros({batch_size, 2, height, width}, depth_src.options());
  auto valid = torch::zeros({batch_size, 1, height, width}, depth_src.options());

  const int count = batch_size * height * width;
  switch (depth_src.scalar_type()) {
    case torch::ScalarType::Double:
      flow_kernel<double>(
        count,
        depth_src.data<double>(),
        depth_tgt.data<double>(),
        height, width,
        KT.data<double>(),
        Kinv.data<double>(),
        flow.data<double>(),
        valid.data<double>()
      );
    case torch::ScalarType::Float:
      flow_kernel<float>(
        count,
        depth_src.data<float>(),
        depth_tgt.data<float>(),
        height, width,
        KT.data<float>(),
        Kinv.data<float>(),
        flow.data<float>(),
        valid.data<float>()
      );
  }  // case
  return {flow, valid};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &flow_cpu_forward, "flow forward (CPU)");
}
