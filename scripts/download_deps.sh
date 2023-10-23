#!/usr/bin/env bash
# some other dependencies
set -x
install=${1:-"all"}

if test "$install" = "all"; then
echo "Installing apt dependencies"
apt-get download -y libjpeg-dev zlib1g-dev
apt-get download -y libopenexr-dev
apt-get download -y openexr
apt-get download -y python3-dev
apt-get download -y libglfw3-dev libglfw3
apt-get download -y libglew-dev
apt-get download -y libassimp-dev
apt-get download -y libnuma-dev  # for byteps
apt download -y clang
## for bop cpp renderer
apt download -y curl
apt download -y autoconf
apt-get download -y build-essential libtool

## for uncertainty pnp
apt-get download -y libeigen3-dev
apt-get download -y libgoogle-glog-dev
apt-get download -y libsuitesparse-dev
apt-get download -y libatlas-base-dev

## for nvdiffrast/egl
apt-get download -y --no-install-recommends \
    cmake curl pkg-config
apt-get download -y --no-install-recommends \
    libgles2 \
    libgl1-mesa-dev \
    libegl1-mesa-dev \
    libgles2-mesa-dev
# (only available for Ubuntu >= 18.04)
apt-get download -y --no-install-recommends \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libglvnd-dev

apt-get download -y libglew-dev
# # for GLEW, add this into ~/.bashrc
# # export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH
fi

# # conda install ipython

# pip install -r requirements/requirements.txt

# # pip install kornia

# pip uninstall pillow
# CC="cc -mavx2" pip install -U --force-reinstall pillow-simd

# # install kaolin

# # (optional) install the nvidia version which is cpp-accelerated
# # git clone https://github.com/NVIDIA/cocoapi.git cocoapi_nvidia
# # cd cocoapi_nvidia/PythonAPI
# # make
# # python setup.py build develop

# # install detectron2
# # git clone https://github.com/facebookresearch/detectron2.git
# # cd detectron2 && pip install -e .

# # install adet  # https://github.com/aim-uofa/adet.git
# # git clone https://github.com/aim-uofa/adet.git
# # cd adet
# # python setup.py build develop
