# -*- coding: utf-8 -*-
import numpy as np
from OpenGL import GL


class Renderbuffer(object):
    def __init__(self, internalformat, W, H):
        self.__id = np.empty(1, dtype=np.uint32)
        GL.glCreateRenderbuffers(len(self.__id), self.__id)
        GL.glNamedRenderbufferStorage(self.__id[0], internalformat, W, H)

    def delete(self):
        GL.glDeleteRenderbuffers(1, self.__id)

    @property
    def id(self):
        return self.__id[0]


class RenderbufferMultisample(object):
    def __init__(self, samples, internalformat, W, H):
        self.__id = np.empty(1, dtype=np.uint32)
        GL.glCreateRenderbuffers(len(self.__id), self.__id)
        GL.glNamedRenderbufferStorageMultisample(self.__id[0], samples, internalformat, W, H)

    def delete(self):
        GL.glDeleteRenderbuffers(1, self.__id)

    @property
    def id(self):
        return self.__id[0]
