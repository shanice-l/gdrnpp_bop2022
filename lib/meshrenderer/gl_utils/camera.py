# -*- coding: utf-8 -*-
# flake8:noqa
import logging as log
from OpenGL import GL
import numpy as np


class Camera(object):
    def __init__(self, offset_u=0.0, offset_v=0.0):
        self.__T_world_view = np.eye(4, dtype=np.float32)
        self.__T_view_world = np.eye(4, dtype=np.float32)

        self.__T_view_proj = np.eye(4, dtype=np.float32)
        self.__T_proj_view = np.eye(4, dtype=np.float32)

        self.__T_proj_world = np.eye(4, dtype=np.float32)

        self.__viewport = (0.0, 0.0, 1.0, 1.0)
        self.__relative_viewport = True
        self._w = 0
        self._h = 0

        self.offset_u = offset_u
        self.offset_v = offset_v

        self.dirty = False

    def lookAt(self, pos, target, up):
        pos = np.array(pos, dtype=np.float32)
        target = np.array(target, dtype=np.float32)
        up = np.array(up, dtype=np.float32)
        z = pos - target
        z *= 1.0 / np.linalg.norm(z)
        x = np.cross(up, z)
        x *= 1.0 / np.linalg.norm(x)
        y = np.cross(z, x)
        rot = np.vstack((x, y, z))
        self.__T_view_world[:3, :3] = rot
        self.__T_view_world[:3, 3] = -(np.dot(rot, pos))
        self.__T_world_view[:3, :3] = rot.transpose()
        self.__T_world_view[:3, 3] = pos
        self.__T_proj_world[:] = np.dot(self.__T_proj_view, self.__T_view_world)
        self.dirty = True

    def from_radius_angles(self, radius, theta, phi):
        x = radius * np.sin(phi) * np.cos(theta)
        y = radius * np.sin(phi) * np.sin(theta)
        z = radius * np.cos(phi)
        pos = np.array((x, y, z), dtype=np.float32)
        target = np.array((0, 0, 0), dtype=np.float32)
        _z = pos - target
        _z *= 1.0 / np.linalg.norm(_z)
        up = (0, 0, 1)
        if np.linalg.norm(np.cross(up, _z)) == 0.0:
            up = (np.cos(theta), np.sin(theta), 0)
        self.lookAt((x, y, z), (0, 0, 0), up)

    def setT_world_view(self, T_world_view):
        self.__T_world_view[:] = T_world_view
        self.__T_view_world[:] = np.linalg.inv(T_world_view)
        self.__T_proj_world[:] = np.dot(self.__T_proj_view, self.__T_view_world)
        self.dirty = True

    def setT_view_proj(self, T_view_proj):
        self.__T_view_proj[:] = T_view_proj
        self.__T_proj_view[:] = np.linalg.inv(T_view_proj)
        self.__T_proj_world[:] = np.dot(self.__T_proj_view, self.__T_view_world)
        self.dirty = True

    def projection(self, fov, aspect, near, far):
        diff = near - far
        A = np.tan(fov / 2.0)
        self.__T_proj_view[:] = np.array(
            [
                [A / aspect, 0, 0, 0],
                [0, A, 0, 0],
                [0, 0, (far + near) / diff, 2 * far * near / diff],
                [0, 0, -1, 0],
            ],
            dtype=np.float32,
        )
        self.__T_view_proj[:] = np.linalg.inv(self.__T_proj_view)
        self.__T_proj_world[:] = np.dot(self.__T_proj_view, self.__T_view_world)
        self.dirty = True

    def realCamera(self, W, H, K, R, t, near, far, scale=1.0, originIsInTopLeft=True):
        self.setIntrinsic(K, W, H, near, far, scale, originIsInTopLeft)
        self.__T_world_view[:3, :3] = R.transpose()
        self.__T_world_view[:3, 3] = -(np.dot(R.transpose(), t.squeeze()))
        z_flip = np.eye(4, dtype=np.float32)
        z_flip[2, 2] = -1
        self.__T_world_view[:] = self.__T_world_view.dot(z_flip)
        self.__T_view_world[:] = np.linalg.pinv(self.__T_world_view)

        self.__T_proj_world[:] = np.dot(self.__T_proj_view, self.__T_view_world)

    def setIntrinsic(self, I, W, H, near, far, scale=1.0, originIsInTopLeft=True):
        """
        Args:
            I:                  3x3 intrinsic camera matrix from real camera (without any OpenGL stuff)
            W:                  Width of the camera image
            H:                  Height of the camera image
            near:               Near plane
            far:                Far plane
            originIsInTopLeft:  If True then the image origin is in top left
                                if False the image origin is in image center

            Source: http://ksimek.github.io/2013/06/03/calibrated_cameras_in_opengl/
        """
        Camera.__check_matrix__(I)

        A = near + far
        B = near * far
        # NOTE: add 0.5 can lead to more accurate PnP/RANSAC result
        u0 = I[0, 2] + self.offset_u  # 0.5
        v0 = I[1, 2] + self.offset_v  # 0.5
        persp = np.array(
            [
                [I[0, 0] * scale, I[0, 1] * scale, -u0 * scale, 0],
                [0, I[1, 1] * scale, -v0 * scale, 0],
                [0, 0, A, B],
                [0, 0, -1, 0],
            ],
            dtype=np.float64,
        )
        ortho = (
            Camera.__glOrtho__(0, W, H, 0, near, far)
            if originIsInTopLeft
            else Camera.__glOrtho__(-W / 2.0, W / 2.0, -H / 2.0, H / 2.0, near, far)
        )

        self.__T_proj_view[:] = np.dot(ortho, persp).astype(np.float32)
        self.__T_view_proj[:] = np.linalg.inv(self.__T_proj_view)
        self.__T_proj_world[:] = np.dot(self.__T_proj_view, self.__T_view_world)
        self.dirty = True

    @staticmethod
    def __check_matrix__(I):
        if len(I.shape) != 2:
            log.error("Camera Matrix not 2D but %dD" % len(I.shape))
            exit(-1)
        elif I.shape != (3, 3):
            log.error("Camera Matrix is not 3x3 but %dx%d" % I.shape)
            exit(-1)
        elif I[1, 0] != 0.0:
            log.error("Camera Matrix Error: Expected Element @ 1,0 to be 0.0 but it's: %.f" % I[1, 0])
            exit(-1)
        elif I[2, 0] != 0.0:
            log.error("Camera Matrix Error: Expected Element @ 2,0 to be 0.0 but it's: %.f" % I[2, 0])
            exit(-1)
        elif I[2, 1] != 0.0:
            log.error("Camera Matrix Error: Expected Element @ 2,1 to be 0.0 but it's: %.f" % I[2, 1])
            exit(-1)
        else:
            pass
        # elif I[2,2] != 1.0:
        #    log.error('Camera Matrix Error: Expected Element @ 2,2 to be 1.0 but it\'s: %.f' % I[2,2])
        #    exit(-1)
        # log.debug('Camera Matrix valid.')

    @staticmethod
    def __glOrtho__(left, right, bottom, top, nearVal, farVal):
        """
        Source: https://www.opengl.org/sdk/docs/man2/xhtml/glOrtho.xhtml
        """
        tx = -(right + left) / (right - left)
        ty = -(top + bottom) / (top - bottom)
        tz = -(farVal + nearVal) / (farVal - nearVal)
        return np.array(
            [
                [2.0 / (right - left), 0.0, 0.0, tx],
                [0.0, 2.0 / (top - bottom), 0.0, ty],
                [0.0, 0.0, -2.0 / (farVal - nearVal), tz],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    @property
    def data(self):
        return np.hstack(
            (
                self.T_view_world.T.reshape(-1),
                self.T_proj_view.T.reshape(-1),
                self.T_world_view[:3, 3].reshape(-1),
            )
        ).astype(np.float32)

    def set_viewport(self, x0, y0, w, h):
        self.__relative_viewport = all([v >= 0.0 and v <= 1.0 for v in [x0, y0, w, h]])
        self.__viewport = (x0, y0, w, h)

    def split_viewport(self, cols, rows, col, row):
        d_r = 1.0 / rows
        d_c = 1.0 / cols
        viewport = (d_c * col, d_r * row, d_c, d_r)
        self.set_viewport(*viewport)

    @property
    def T_world_view(self):
        return self.__T_world_view

    @property
    def T_view_world(self):
        return self.__T_view_world

    @property
    def T_view_proj(self):
        return self.__T_view_proj

    @property
    def T_proj_view(self):
        return self.__T_proj_view

    @property
    def T_proj_world(self):
        return self.__T_proj_world

    def get_viewport(self):
        v = self.__viewport
        if self.__relative_viewport:
            W, H = self._w, self._h
            return (int(v[0] * W), int(v[1] * H), int(v[2] * W), int(v[3] * H))
        else:
            return v
