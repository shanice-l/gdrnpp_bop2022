# -*- coding: utf-8 -*-
import numpy as np
from OpenGL import GL


class EBO(object):
    def __init__(self, data, dynamic=False):
        self.__id = np.empty(1, dtype=np.uint32)
        GL.glCreateBuffers(len(self.__id), self.__id)
        code = 0 if not dynamic else GL.GL_DYNAMIC_STORAGE_BIT | GL.GL_MAP_WRITE_BIT | GL.GL_MAP_PERSISTENT_BIT
        GL.glNamedBufferStorage(self.__id, data.nbytes, data, code)

    def bind(self):
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.__id)

    @property
    def id(self):
        return self.__id
