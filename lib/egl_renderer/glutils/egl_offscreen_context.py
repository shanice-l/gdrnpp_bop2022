# -*- coding: utf-8 -*-
# flake8: noqa
import ctypes
import os

if not os.environ.get("PYOPENGL_PLATFORM"):
    os.environ["PYOPENGL_PLATFORM"] = "egl"

import OpenGL.platform

_GPU_ID = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
EGL_PLATFORM_DEVICE_EXT = 0x313F
EGL_DRM_DEVICE_FILE_EXT = 0x3233


def _ensure_egl_loaded():
    plugin = OpenGL.platform.PlatformPlugin.by_name("egl")
    if plugin is None:
        raise RuntimeError("EGL platform plugin is not available.")

    plugin_class = plugin.load()
    plugin.loaded = True
    # create instance of this platform implementation
    plugin = plugin_class()

    plugin.install(vars(OpenGL.platform))


_ensure_egl_loaded()
from OpenGL import EGL as egl

from OpenGL.EGL import (
    EGL_SURFACE_TYPE,
    EGL_PBUFFER_BIT,
    EGL_BLUE_SIZE,
    EGL_RED_SIZE,
    EGL_GREEN_SIZE,
    EGL_DEPTH_SIZE,
    EGL_COLOR_BUFFER_TYPE,
    EGL_RGB_BUFFER,
    EGL_RENDERABLE_TYPE,
    EGL_OPENGL_BIT,
    EGL_CONFORMANT,
    EGL_NONE,
    EGL_DEFAULT_DISPLAY,
    EGL_NO_CONTEXT,
    EGL_OPENGL_API,
    EGL_CONTEXT_MAJOR_VERSION,
    EGL_CONTEXT_MINOR_VERSION,
    EGL_CONTEXT_OPENGL_PROFILE_MASK,
    EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT,
    eglGetDisplay,
    eglInitialize,
    eglChooseConfig,
    eglBindAPI,
    eglCreateContext,
    EGLConfig,
)
from OpenGL import arrays
from OpenGL.GL.NV.bindless_texture import *

__all__ = ["OffscreenContext"]


def _get_egl_func(func_name, res_type, *arg_types):
    address = egl.eglGetProcAddress(func_name)
    if address is None:
        return None

    proto = ctypes.CFUNCTYPE(res_type)
    proto.argtypes = arg_types
    func = proto(address)
    return func


def _get_egl_struct(struct_name):
    from OpenGL._opaque import opaque_pointer_cls

    return opaque_pointer_cls(struct_name)


# These are not defined in PyOpenGL by default.
_EGLDeviceEXT = _get_egl_struct("EGLDeviceEXT")
_eglGetPlatformDisplayEXT = _get_egl_func("eglGetPlatformDisplayEXT", egl.EGLDisplay)
_eglQueryDevicesEXT = _get_egl_func("eglQueryDevicesEXT", egl.EGLBoolean)
_eglQueryDeviceStringEXT = _get_egl_func("eglQueryDeviceStringEXT", ctypes.c_char_p)


def query_devices():
    if _eglQueryDevicesEXT is None:
        raise RuntimeError("EGL query extension is not loaded or is not supported.")

    num_devices = egl.EGLint()
    success = _eglQueryDevicesEXT(0, None, ctypes.pointer(num_devices))
    if not success or num_devices.value < 1:
        return []

    devices = (_EGLDeviceEXT * num_devices.value)()  # array of size num_devices
    success = _eglQueryDevicesEXT(num_devices.value, devices, ctypes.pointer(num_devices))
    if not success or num_devices.value < 1:
        return []

    return [EGLDevice(devices[i]) for i in range(num_devices.value)]


def get_default_device():
    # Fall back to not using query extension.
    if _eglQueryDevicesEXT is None:
        return EGLDevice(None)

    return query_devices()[0]


def get_device_by_index(device_id):
    if _eglQueryDevicesEXT is None and device_id == 0:
        return get_default_device()

    devices = query_devices()
    if device_id >= len(devices):
        raise ValueError("Invalid device ID ({})".format(device_id, len(devices)))
    return devices[device_id]


class EGLDevice:
    def __init__(self, display=None):
        self._display = display

    def get_display(self):
        if self._display is None:
            return egl.eglGetDisplay(egl.EGL_DEFAULT_DISPLAY)

        return _eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT, self._display, None)

    @property
    def name(self):
        if self._display is None:
            return "default"

        name = _eglQueryDeviceStringEXT(self._display, EGL_DRM_DEVICE_FILE_EXT)
        if name is None:
            return None

        return name.decode("ascii")

    def __repr__(self):
        return "<EGLDevice(name={})>".format(self.name)


class OffscreenContext(object):
    def __init__(self, gpu_id=None):
        if gpu_id is None or gpu_id < 0:
            gpu_id = _GPU_ID
        device = get_device_by_index(gpu_id)
        self._egl_device = device
        self._egl_display = None
        self._egl_context = None
        self.init_context()

    def init_context(self):
        _ensure_egl_loaded()
        config_attributes = arrays.GLintArray.asArray(
            [
                EGL_BLUE_SIZE,
                8,
                EGL_RED_SIZE,
                8,
                EGL_GREEN_SIZE,
                8,
                EGL_DEPTH_SIZE,
                24,
                EGL_SURFACE_TYPE,
                EGL_PBUFFER_BIT,
                EGL_COLOR_BUFFER_TYPE,
                EGL_RGB_BUFFER,
                EGL_RENDERABLE_TYPE,
                EGL_OPENGL_BIT,
                EGL_CONFORMANT,
                EGL_OPENGL_BIT,
                EGL_NONE,
            ]
        )
        context_attributes = arrays.GLintArray.asArray(
            [
                EGL_CONTEXT_MAJOR_VERSION,
                4,
                EGL_CONTEXT_MINOR_VERSION,
                1,
                EGL_CONTEXT_OPENGL_PROFILE_MASK,
                EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT,
                EGL_NONE,
            ]
        )
        major, minor = ctypes.c_long(), ctypes.c_long()
        num_configs = ctypes.c_long()
        configs = (EGLConfig * 1)()

        # Cache DISPLAY if necessary and get an off-screen EGL display
        orig_dpy = None
        if "DISPLAY" in os.environ:
            orig_dpy = os.environ["DISPLAY"]
            del os.environ["DISPLAY"]
        # self._egl_display = eglGetDisplay(EGL_DEFAULT_DISPLAY)
        self._egl_display = self._egl_device.get_display()
        if orig_dpy is not None:
            os.environ["DISPLAY"] = orig_dpy

        # Initialize EGL
        assert eglInitialize(self._egl_display, major, minor)
        assert eglChooseConfig(self._egl_display, config_attributes, configs, 1, num_configs)
        # assert eglChooseConfig(
        #     self._egl_display, config_attributes, ctypes.pointer(configs), 1, ctypes.pointer(num_configs)
        # )

        # Bind EGL to the OpenGL API
        assert eglBindAPI(EGL_OPENGL_API)

        # Create an EGL context
        self._egl_context = eglCreateContext(self._egl_display, configs[0], EGL_NO_CONTEXT, context_attributes)
        if self._egl_context == EGL_NO_CONTEXT:
            raise RuntimeError("Unable to create context")

        # Make it current
        self.make_current()

        if not glInitBindlessTextureNV():
            raise RuntimeError("Bindless Textures not supported")
        self.__display = self._egl_display

    def make_current(self):
        from OpenGL.EGL import eglMakeCurrent, EGL_NO_SURFACE

        assert eglMakeCurrent(
            self._egl_display,
            EGL_NO_SURFACE,
            EGL_NO_SURFACE,
            self._egl_context,
        )

    def make_uncurrent(self):
        """Make the OpenGL context uncurrent."""
        pass

    def close(self):
        self.delete_context()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def delete_context(self):
        from OpenGL.EGL import eglDestroyContext, eglTerminate

        if self._egl_display is not None:
            if self._egl_context is not None:
                eglDestroyContext(self._egl_display, self._egl_context)
                self._egl_context = None
            eglTerminate(self._egl_display)
            self._egl_display = None

    def supports_framebuffers(self):
        return True


if __name__ == "__main__":
    conntext = OffscreenContext()
