# -*- coding: utf-8 -*-
# flake8: noqa
import numpy as np
from OpenGL.GL import *
from OpenGL.GL.NV.bindless_texture import *


class Texture(object):
    def __init__(self, tex_type, levels, internalformat, W, H):
        self.__id = np.empty(1, dtype=np.uint32)
        glCreateTextures(tex_type, len(self.__id), self.__id)
        glTextureStorage2D(self.__id[0], levels, internalformat, W, H)
        self.__handle = None

    def setFilter(self, min_filter, max_filter):
        glTextureParameteri(self.__id[0], GL_TEXTURE_MIN_FILTER, min_filter)
        glTextureParameteri(self.__id[0], GL_TEXTURE_MAG_FILTER, max_filter)

    def setWrap(self, wrap_s, wrap_t, wrap_r=None):
        glTextureParameteri(self.__id[0], GL_TEXTURE_WRAP_S, wrap_s)
        glTextureParameteri(self.__id[0], GL_TEXTURE_WRAP_T, wrap_t)
        if wrap_r != None:
            glTextureParameteri(self.__id[0], GL_TEXTURE_WRAP_R, wrap_r)

    def subImage(
        self,
        level,
        xoffset,
        yoffset,
        width,
        height,
        data_format,
        data_type,
        pixels,
    ):
        glTextureSubImage2D(
            self.__id[0],
            level,
            xoffset,
            yoffset,
            width,
            height,
            data_format,
            data_type,
            pixels,
        )

    def generate_mipmap(self):
        glGenerateTextureMipmap(self.__id[0])

    def makeResident(self):
        self.__handle = glGetTextureHandleNV(self.__id[0])
        glMakeTextureHandleResidentNV(self.__handle)
        return self.__handle

    def makeNonResident(self):
        if self.__handle != None:
            glMakeTextureHandleNonResidentNV(self.__handle)

    def delete(self):
        glDeleteTextures(1, self.__id)

    @property
    def handle(self):
        return self.__handle

    @property
    def id(self):
        return self.__id[0]


class Texture1D(object):
    def __init__(self, levels, internalformat, W):
        self.__id = np.empty(1, dtype=np.uint32)
        glCreateTextures(GL_TEXTURE_1D, len(self.__id), self.__id)
        glTextureStorage1D(self.__id[0], levels, internalformat, W)
        self.__handle = None

    def setFilter(self, min_filter, max_filter):
        glTextureParameteri(self.__id[0], GL_TEXTURE_MIN_FILTER, min_filter)
        glTextureParameteri(self.__id[0], GL_TEXTURE_MAG_FILTER, max_filter)

    def setWrap(self, wrap_s, wrap_t, wrap_r=None):
        glTextureParameteri(self.__id[0], GL_TEXTURE_WRAP_S, wrap_s)
        glTextureParameteri(self.__id[0], GL_TEXTURE_WRAP_T, wrap_t)
        if wrap_r != None:
            glTextureParameteri(self.__id[0], GL_TEXTURE_WRAP_R, wrap_r)

    def subImage(self, level, xoffset, width, data_format, data_type, pixels):
        glTextureSubImage1D(self.__id[0], level, xoffset, width, data_format, data_type, pixels)

    def generate_mipmap(self):
        glGenerateTextureMipmap(self.__id[0])

    def makeResident(self):
        self.__handle = glGetTextureHandleNV(self.__id[0])
        glMakeTextureHandleResidentNV(self.__handle)
        return self.__handle

    def makeNonResident(self):
        if self.__handle != None:
            glMakeTextureHandleNonResidentNV(self.__handle)

    def delete(self):
        glDeleteTextures(1, self.__id)

    @property
    def handle(self):
        return self.__handle

    @property
    def id(self):
        return self.__id[0]


class Texture3D(object):
    def __init__(self, tex_type, levels, internalformat, W, H, C):
        self.__id = np.empty(1, dtype=np.uint32)
        glCreateTextures(tex_type, len(self.__id), self.__id)
        glTextureStorage3D(self.__id[0], levels, internalformat, W, H, C)
        self.__handle = None

    def setFilter(self, min_filter, max_filter):
        glTextureParameteri(self.__id[0], GL_TEXTURE_MIN_FILTER, min_filter)
        glTextureParameteri(self.__id[0], GL_TEXTURE_MAG_FILTER, max_filter)

    def setWrap(self, wrap_s, wrap_t, wrap_r=None):
        glTextureParameteri(self.__id[0], GL_TEXTURE_WRAP_S, wrap_s)
        glTextureParameteri(self.__id[0], GL_TEXTURE_WRAP_T, wrap_t)
        if wrap_r != None:
            glTextureParameteri(self.__id[0], GL_TEXTURE_WRAP_R, wrap_r)

    def subImage(
        self,
        level,
        xoffset,
        yoffset,
        zoffset,
        width,
        height,
        depth,
        data_format,
        data_type,
        pixels,
    ):
        glTextureSubImage3D(
            self.__id[0],
            level,
            xoffset,
            yoffset,
            zoffset,
            width,
            height,
            depth,
            data_format,
            data_type,
            pixels,
        )

    def generate_mipmap(self):
        glGenerateTextureMipmap(self.__id[0])

    def makeResident(self):
        self.__handle = glGetTextureHandleNV(self.__id[0])
        glMakeTextureHandleResidentNV(self.__handle)
        return self.__handle

    def makeNonResident(self):
        if self.__handle != None:
            glMakeTextureHandleNonResidentNV(self.__handle)

    def delete(self):
        glDeleteTextures(1, self.__id)

    @property
    def handle(self):
        return self.__handle

    @property
    def id(self):
        return self.__id[0]


class TextureMultisample(object):
    def __init__(self, samples, internalformat, W, H, fixedsamplelocations):
        self.__id = np.empty(1, dtype=np.uint32)
        glCreateTextures(GL_TEXTURE_2D_MULTISAMPLE, len(self.__id), self.__id)
        glTextureStorage2DMultisample(self.__id[0], samples, internalformat, W, H, fixedsamplelocations)
        self.__handle = None

    def setFilter(self, min_filter, max_filter):
        glTextureParameteri(self.__id[0], GL_TEXTURE_MIN_FILTER, min_filter)
        glTextureParameteri(self.__id[0], GL_TEXTURE_MAG_FILTER, max_filter)

    def setWrap(self, wrap_s, wrap_t):
        glTextureParameteri(self.__id[0], GL_TEXTURE_WRAP_S, wrap_s)
        glTextureParameteri(self.__id[0], GL_TEXTURE_WRAP_T, wrap_t)

    def subImage(
        self,
        level,
        xoffset,
        yoffset,
        width,
        height,
        data_format,
        data_type,
        pixels,
    ):
        glTextureSubImage2D(
            self.__id[0],
            level,
            xoffset,
            yoffset,
            width,
            height,
            data_format,
            data_type,
            pixels,
        )

    def makeResident(self):
        self.__handle = glGetTextureHandleNV(self.__id[0])
        glMakeTextureHandleResidentNV(self.__handle)
        return self.__handle

    def makeNonResident(self):
        if self.__handle != None:
            glMakeTextureHandleNonResidentNV(self.__handle)

    def delete(self):
        glDeleteTextures(1, self.__id)

    @property
    def handle(self):
        return self.__handle

    @property
    def id(self):
        return self.__id[0]


def loadTexture(path):
    from PIL import Image
    import OpenGL.GL as GL

    img = Image.open(path).transpose(Image.FLIP_TOP_BOTTOM)
    if img.mode != "RGB":
        # print('convert {} to RGB'.format(img.mode))
        img = img.convert("RGB")
    img_data = np.fromstring(img.tobytes(), np.uint8)
    # print(img_data.shape)
    width, height = img.size
    # glTexImage2D expects the first element of the image data to be the
    # bottom-left corner of the image.  Subsequent elements go left to right,
    # with subsequent lines going from bottom to top.

    # However, the image data was created with PIL Image tostring and numpy's
    # fromstring, which means we have to do a bit of reorganization. The first
    # element in the data output by tostring() will be the top-left corner of
    # the image, with following values going left-to-right and lines going
    # top-to-bottom.  So, we need to flip the vertical coordinate (y).
    texture_id = GL.glGenTextures(1)
    GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, texture_id)  # bind texture
    GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
    GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR_MIPMAP_LINEAR)
    GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
    GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
    GL.glTexImage2D(
        GL.GL_TEXTURE_2D,
        0,
        GL.GL_RGB,
        width,
        height,
        0,
        GL.GL_RGB,
        GL.GL_UNSIGNED_BYTE,
        img_data,
    )
    GL.glGenerateMipmap(GL.GL_TEXTURE_2D)
    return texture_id
