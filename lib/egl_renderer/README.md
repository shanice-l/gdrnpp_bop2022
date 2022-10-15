# EGL Renderer

This is an EGL version of PyOpenGL renderer, it has some key features to work with deep learning frameworks.
- EGL headless rendering, so a running x server is not required.
- Render rgb, segmentation, point cloud in a single shader call by rendering to multiple texture buffers.
- Render to pytorch tensor directly via OpenGL - CUDA interoperation.
- Enable choosing which GPU to use via `eglQueryDevicesEXT` and `eglGetPlatformDisplayEXT`.
- Minimalistic, it uses raw OpenGL, doesn't rely on third-party GL helper libraries other than `assimp`.

## Installation
### Install dependencies:

```bash
# 1) you need to install nvidia OpenGL drivers (https://www.nvidia.com/en-us/drivers/unix/) and make them visible to find libEGL.so
# /usr/lib/x86_64-linux-gnu/libEGL.so
# 2) Install other dependencies like PyOpenGL to locate EGL. (uninstall other pyopengl first)
sh scripts/install_deps.sh
```
### Compile:
In a virtual environment, and in the folder `lib/egl_renderer/`, run
```bash
sh compile_cpp_egl_renderer.sh
```

To check whether it is successfully built, run `./lib/egl_renderer/build/query_devices`

### Optional
If you want to render to pytorch tensor, you need to install [pytorch](https://pytorch.org).

If you want to render to pycuda tensor, you need to install [pycuda](https://documen.tician.de/pycuda/) and compile pycuda with GL suport.

## Example
`python -m lib.egl_renderer.egl_renderer_v3`
Look details in this file to run other tests.
