import os
from os.path import join as pjoin

from setuptools import setup
from torch.utils.cpp_extension import (
    BuildExtension,
    CppExtension,
    CUDAExtension,
)


def find_in_path(name, path):
    """Find a file in a search path."""
    # adapted fom http://code.activestate.com/recipes/52224-find-a-file-given-a-search-path/
    for dir in path.split(os.pathsep):
        binpath = pjoin(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None


def locate_cuda():
    """Locate the CUDA environment on the system.

    Returns a dict with keys 'home', 'nvcc', 'include', and 'lib64'
    and values giving the absolute path to each directory.

    Starts by looking for the CUDA_HOME env variable. If not found, everything
    is based on finding 'nvcc' in the PATH.
    """

    # first check if the CUDA_HOME env variable is in use
    if "CUDA_HOME" in os.environ:
        home = os.environ["CUDA_HOME"]
        nvcc = pjoin(home, "bin", "nvcc")
    else:
        # otherwise, search the PATH for NVCC
        nvcc = find_in_path("nvcc", os.environ["PATH"])
        if nvcc is None:
            raise EnvironmentError(
                "The nvcc binary could not be " "located in your $PATH. Either add it to your path, or set $CUDA_HOME"
            )
        home = os.path.dirname(os.path.dirname(nvcc))

    cudaconfig = {
        "home": home,
        "nvcc": nvcc,
        "include": pjoin(home, "include"),
        "lib64": pjoin(home, "lib64"),
    }
    for k, v in cudaconfig.items():
        if not os.path.exists(v):
            raise EnvironmentError("The CUDA %s path could not be located in %s" % (k, v))

    return cudaconfig


CUDA = locate_cuda()

setup(
    name="torch_nndistance",
    ext_modules=[
        CUDAExtension(
            "torch_nndistance_aten",
            ["src/nnd_cuda.cpp", "src/nnd_cuda_kernel.cu"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    packages=["torch_nndistance"],
    classifiers=["Programming Language :: Python :: 3"],
)
