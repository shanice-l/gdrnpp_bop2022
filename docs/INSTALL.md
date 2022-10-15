# Installation

* CUDA >= 10.1, Ubuntu >= 16.04

* Python >= 3.6, PyTorch >= 1.9, torchvision
    ```
    ## install Anaconda or Minoconda
    # If in THU, set conda and pip source to tuna
    # setting .condarc: https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/
    # setting pypi:
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

    ## create a new environment
    conda create -n py37 python=3.7.4
    conda activate py37  # maybe add this line to the end of ~/.bashrc
    conda install ipython
    ## install pytorch: https://pytorch.org/get-started/locally/  (check cuda version)
    pip install torchvision -U  # will also install corresponding torch
    ```

* `detectron2` from [source](https://github.com/facebookresearch/detectron2).
    ```
    git clone https://github.com/facebookresearch/detectron2.git
    cd detectron2
    pip install ninja
    pip install -e .
    ```

* `sh scripts/install_deps.sh`
    * if you are not a sudoer, you can just install the python packages:
      ```
      sh scripts/install_deps.sh python
      ```
    * or ask the sudoer to install the apt packages (usually already installed)

* NOTE: for Ubuntu 18.04, please refer to [INSTALL_libassimp_ubuntu18_04](INSTALL_libassimp_ubuntu18_04.md) to install `libassimp-dev 3.2`.
    ```
    # check Ubuntu version:
    lsb_release -a
    ```

* cpp/cuda extensions:
    * build all extensions by `sh scripts/compile_all.sh` (for uncertainty_pnp on Ubuntu 18.04, need to build ceres first, see below) or

    * build each extension separately
        * farthest points sampling
        ```
        cd core/csrc
        sh compile_fps.sh
        ```

        * egl_renderer (directly render to torch cuda tensors)
        ```
        cd lib/egl_renderer
        sh compile_cpp_egl_renderer.sh
        ```

        * flow_cuda (for deepim)
        ```
        cd core/csrc/
        sh compile_flow.sh
        ```

            * If failed on Ubuntu 16.04 with gcc 5, try installing gcc-7 and g++-7
            ```
            sudo apt-get install -y software-properties-common
            sudo add-apt-repository ppa:ubuntu-toolchain-r/test
            sudo apt update
            sudo apt install g++-7 -y
            ```
            Then run
            ```
            CC=gcc-7 CXX=g++-7 sh compile_flow.sh
            ```

        * ransac voting (for pvnet)
        ```
        cd core/csrc/
        sh compile_ransac_voting.sh
        ```

        * uncertainty pnp (for pvnet)
        ```
        cd core/csrc/
        cd uncertainty_pnp/
        # cd lib; ln -sf libceres.so.2 libceres.so; cd .. # (Ubuntu 16.04)
        # sh build_ceres.sh  # (Ubuntu >= 18.04)
        cd ..
        sh compile_uncertainty_pnp.sh
        ```

        * torch nndistance (chamfer distance)
        ```
        cd core/csrc/
        sh compile_nndistance.sh
        ```
        **NOTE**: Run `. scripts/init_env.sh` to use uncertainty_pnp.
