#!/usr/bin/env bash
# some other dependencies
set -x
install=${1:-"all"}

if test "$install" = "all"; then
echo "Installing apt dependencies"
sudo apt-get install -y libjpeg-dev zlib1g-dev
sudo apt-get install -y libopenexr-dev
sudo apt-get install -y openexr
sudo apt-get install -y python3-dev
sudo apt-get install -y libglfw3-dev libglfw3
sudo apt-get install -y libglew-dev
sudo apt-get install -y libassimp-dev
sudo apt-get install -y libnuma-dev  # for byteps
sudo apt install -y clang
## for bop cpp renderer
sudo apt install -y curl
sudo apt install -y autoconf
sudo apt-get install -y build-essential libtool

## for uncertainty pnp
sudo apt-get install -y libeigen3-dev
sudo apt-get install -y libgoogle-glog-dev
sudo apt-get install -y libsuitesparse-dev
sudo apt-get install -y libatlas-base-dev

## for nvdiffrast/egl
sudo apt-get install -y --no-install-recommends \
    cmake curl pkg-config
sudo apt-get install -y --no-install-recommends \
    libgles2 \
    libgl1-mesa-dev \
    libegl1-mesa-dev \
    libgles2-mesa-dev
# (only available for Ubuntu >= 18.04)
sudo apt-get install -y --no-install-recommends \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libglvnd-dev

sudo apt-get install -y libglew-dev
# for GLEW, add this into ~/.bashrc
# export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH
fi

# conda install ipython

# pip install -r requirements/requirements.txt

# pip install kornia

# pip uninstall pillow
# CC="cc -mavx2" pip install -U --force-reinstall pillow-simd

# install kaolin

# (optional) install the nvidia version which is cpp-accelerated
# git clone https://github.com/NVIDIA/cocoapi.git cocoapi_nvidia
# cd cocoapi_nvidia/PythonAPI
# make
# python setup.py build develop

# install detectron2
# git clone https://github.com/facebookresearch/detectron2.git
# cd detectron2 && pip install -e .

# install adet  # https://github.com/aim-uofa/adet.git
# git clone https://github.com/aim-uofa/adet.git
# cd adet
# python setup.py build develop
