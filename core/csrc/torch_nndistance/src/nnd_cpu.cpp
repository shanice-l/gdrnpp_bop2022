#include <torch/torch.h>

void nnsearch(int b,int n,int m,const float * xyz1,const float * xyz2,float * dist,int * idx){
    for (int i=0;i<b;i++){
        for (int j=0;j<n;j++){
            float x1=xyz1[(i*n+j)*3+0];
            float y1=xyz1[(i*n+j)*3+1];
            float z1=xyz1[(i*n+j)*3+2];
            double best=0;
            int besti=0;
            for (int k=0;k<m;k++){
                float x2=xyz2[(i*m+k)*3+0]-x1;
                float y2=xyz2[(i*m+k)*3+1]-y1;
                float z2=xyz2[(i*m+k)*3+2]-z1;
                double d=x2*x2+y2*y2+z2*z2;
                if (k==0 || d<best){
                    best=d;
                    besti=k;
                }
            }
            dist[i*n+j]=best;
            idx[i*n+j]=besti;
        }
    }
}


int nnd_forward(
    at::Tensor xyz1,
    at::Tensor xyz2,
    at::Tensor dist1,
    at::Tensor dist2,
    at::Tensor idx1,
    at::Tensor idx2)
{
    int batchsize = xyz1.size(0);
    int n = xyz1.size(1);
    int m = xyz2.size(1);

    //printf("%d %d %d\n", batchsize, n, m);
/*    auto dist1 = at::zeros(at::IntList({batchsize, n}));
    auto dist2 = at::zeros(at::IntList({batchsize, n}));
    auto idx1  = at::zeros(at::IntList({batchsize, n}));
    auto idx2  = at::zeros(at::IntList({batchsize, n}));
    idx1.toType(c10::ScalarType::Int);
    idx2.toType(c10::ScalarType::Int);
    dist1.toType(c10::ScalarType::Float);
    dist2.toType(c10::ScalarType::Float);*/

    float *xyz1_data = xyz1.data_ptr<float>();
    float *xyz2_data = xyz2.data_ptr<float>();
    float *dist1_data = dist1.data_ptr<float>();
    float *dist2_data = dist2.data_ptr<float>();
    int *idx1_data = idx1.data_ptr<int>();
    int *idx2_data = idx2.data_ptr<int>();
    nnsearch(batchsize, n, m, xyz1_data, xyz2_data, dist1_data, idx1_data);
    nnsearch(batchsize, m, n, xyz2_data, xyz1_data, dist2_data, idx2_data);

    return 1;
}



int nnd_backward(
    at::Tensor xyz1,
    at::Tensor xyz2,
    at::Tensor gradxyz1,
    at::Tensor gradxyz2,
    at::Tensor graddist1,
    at::Tensor graddist2,
    at::Tensor idx1,
    at::Tensor idx2)
{
    int b = xyz1.size(0);
    int n = xyz1.size(1);
    int m = xyz2.size(1);

/*
    auto gradxyz1 = at::zeros_like(xyz1);
    auto gradxyz2 = at::zeros_like(xyz2);
    */
    float *xyz1_data = xyz1.data_ptr<float>();
    float *xyz2_data = xyz2.data_ptr<float>();
    float *gradxyz1_data = gradxyz1.data_ptr<float>();
    float *gradxyz2_data = gradxyz2.data_ptr<float>();
    float *graddist1_data = graddist1.data_ptr<float>();
    float *graddist2_data = graddist2.data_ptr<float>();
    int *idx1_data = idx1.data_ptr<int>();
    int *idx2_data = idx2.data_ptr<int>();

    for (int i=0;i<b*n*3;i++)
        gradxyz1_data[i]=0;
    for (int i=0;i<b*m*3;i++)
        gradxyz2_data[i]=0;
    for (int i=0;i<b;i++){
        for (int j=0;j<n;j++){
            float x1=xyz1_data[(i*n+j)*3+0];
            float y1=xyz1_data[(i*n+j)*3+1];
            float z1=xyz1_data[(i*n+j)*3+2];
            int j2=idx1_data[i*n+j];

            float x2=xyz2_data[(i*m+j2)*3+0];
            float y2=xyz2_data[(i*m+j2)*3+1];
            float z2=xyz2_data[(i*m+j2)*3+2];
            float g=graddist1_data[i*n+j]*2;

            gradxyz1_data[(i*n+j)*3+0]+=g*(x1-x2);
            gradxyz1_data[(i*n+j)*3+1]+=g*(y1-y2);
            gradxyz1_data[(i*n+j)*3+2]+=g*(z1-z2);
            gradxyz2_data[(i*m+j2)*3+0]-=(g*(x1-x2));
            gradxyz2_data[(i*m+j2)*3+1]-=(g*(y1-y2));
            gradxyz2_data[(i*m+j2)*3+2]-=(g*(z1-z2));
        }
        for (int j=0;j<m;j++){
            float x1=xyz2_data[(i*m+j)*3+0];
            float y1=xyz2_data[(i*m+j)*3+1];
            float z1=xyz2_data[(i*m+j)*3+2];
            int j2=idx2_data[i*m+j];
            float x2=xyz1_data[(i*n+j2)*3+0];
            float y2=xyz1_data[(i*n+j2)*3+1];
            float z2=xyz1_data[(i*n+j2)*3+2];
            float g=graddist2_data[i*m+j]*2;
            gradxyz2_data[(i*m+j)*3+0]+=g*(x1-x2);
            gradxyz2_data[(i*m+j)*3+1]+=g*(y1-y2);
            gradxyz2_data[(i*m+j)*3+2]+=g*(z1-z2);
            gradxyz1_data[(i*n+j2)*3+0]-=(g*(x1-x2));
            gradxyz1_data[(i*n+j2)*3+1]-=(g*(y1-y2));
            gradxyz1_data[(i*n+j2)*3+2]-=(g*(z1-z2));
        }
    }

    return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("nnd_forward", &nnd_forward, "nnd_forward");
  m.def("nnd_backward", &nnd_backward, "nnd_backward");
}
