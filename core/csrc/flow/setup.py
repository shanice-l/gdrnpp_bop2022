import os
import platform
import subprocess
import time
from setuptools import Extension, find_packages, setup

import numpy as np
from Cython.Build import cythonize
from torch.utils.cpp_extension import (
    BuildExtension,
    CUDAExtension,
    CppExtension,
)


def make_cuda_ext(name, module, sources):

    return CUDAExtension(
        name="{}.{}".format(module, name),
        sources=[os.path.join(*module.split("."), p) for p in sources],
        extra_compile_args={
            "cxx": [],
            "nvcc": [
                "-D__CUDA_NO_HALF_OPERATORS__",
                "-D__CUDA_NO_HALF_CONVERSIONS__",
                "-D__CUDA_NO_HALF2_OPERATORS__",
            ],
        },
    )


def make_cpu_ext(name, module, sources):

    return CppExtension(
        name="{}.{}".format(module, name),
        sources=[os.path.join(*module.split("."), p) for p in sources],
        extra_compile_args={"cxx": []},
    )


def make_cython_ext(name, module, sources):
    extra_compile_args = None
    if platform.system() != "Windows":
        extra_compile_args = {
            "cxx": [
                "-Wno-unused-function",
                "-Wno-write-strings",
                "-ffast-math",
            ]
        }

    extension = Extension(
        "{}.{}".format(module, name),
        [os.path.join(*module.split("."), p) for p in sources],
        include_dirs=[np.get_include()],
        language="c++",
        extra_compile_args=extra_compile_args,
    )
    (extension,) = cythonize(extension)
    return extension


if __name__ == "__main__":
    setup(
        description="flow_cuda",
        url="none",
        setup_requires=["cython", "numpy"],
        ext_modules=[
            make_cuda_ext(
                name="flow_cuda",
                module=".",
                sources=["src/flow_cuda_kernel.cu", "src/flow_cuda.cpp"],
            )
        ],
        cmdclass={"build_ext": BuildExtension},
        zip_safe=False,
    )
