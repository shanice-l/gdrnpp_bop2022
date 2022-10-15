# -*- coding: utf-8 -*-
import numpy as np
from OpenGL import GL


class IBO(object):
    def __init__(
        self,
        sizes,
        instances,
        offsets,
        base_instance,
        first_index=None,
        dynamic=False,
    ):
        if not isinstance(first_index, np.ndarray):
            indices = np.vstack((sizes, instances, offsets, base_instance)).T.astype(np.uint32).copy()
        else:
            indices = np.vstack((sizes, instances, offsets, base_instance, first_index)).T.astype(np.uint32).copy()
        self.__id = np.empty(1, dtype=np.uint32)
        GL.glCreateBuffers(len(self.__id), self.__id)
        code = 0 if not dynamic else GL.GL_DYNAMIC_STORAGE_BIT | GL.GL_MAP_WRITE_BIT | GL.GL_MAP_PERSISTENT_BIT
        GL.glNamedBufferStorage(self.__id, indices.nbytes, indices, code)

    def bind(self):
        GL.glBindBuffer(GL.GL_DRAW_INDIRECT_BUFFER, self.__id)
