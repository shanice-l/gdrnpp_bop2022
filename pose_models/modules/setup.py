from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='corr_sampler',
    ext_modules=[
        CUDAExtension('corr_sampler', 
            sources=[
                'extensions/sampler.cpp', 
                'extensions/sampler_kernel.cu',
            ],
            extra_compile_args={
                'cxx': ['-O2'],
                'nvcc': ['-O2',
                    '-arch=sm_50',
                    '-gencode=arch=compute_37,code=sm_37',
                    '-gencode=arch=compute_50,code=sm_50',
                    '-gencode=arch=compute_52,code=sm_52',
                    '-gencode=arch=compute_60,code=sm_60',
                    '-gencode=arch=compute_61,code=sm_61',
                    '-gencode=arch=compute_70,code=sm_70',
                    '-gencode=arch=compute_75,code=sm_75',
                    '-gencode=arch=compute_75,code=compute_75',

                ]
            }),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

