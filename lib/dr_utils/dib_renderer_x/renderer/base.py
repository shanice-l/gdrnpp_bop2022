# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from __future__ import print_function
from __future__ import division

from core.utils.pose_utils import quat2mat_torch

from ..utils import perspectiveprojectionnp, projectiveprojection_real
from .phongrender import PhongRender
from .shrender import SHRender
from .texrender import TexRender as Lambertian
from .vcrender import VCRender
from .vcrender_batch import VCRenderBatch
from .vcrender_multi import VCRenderMulti
from .texrender_multi import TexRenderMulti
from .texrender_batch import TexRenderBatch
import numpy as np
import torch
import torch.nn as nn

# renderers = {'VertexColor': VCRender, 'Lambertian': Lambertian, 'SphericalHarmonics': SHRender, 'Phong': PhongRender}
renderers = {
    "VertexColor": VCRender,
    "VertexColorMulti": VCRenderMulti,
    "VertexColorBatch": VCRenderBatch,
    "Lambertian": Lambertian,
    "Texture": Lambertian,  # alias
    "TextureMulti": TexRenderMulti,
    "TextureBatch": TexRenderBatch,
    "SphericalHarmonics": SHRender,
    "Phong": PhongRender,
}


class Renderer(nn.Module):
    def __init__(
        self,
        height,
        width,
        mode="VertexColor",
        camera_center=None,
        camera_up=None,
        camera_fov_y=None,
    ):
        super(Renderer, self).__init__()
        assert mode in renderers, "Passed mode {0} must in in list of accepted modes: {1}".format(mode, renderers)
        self.mode = mode

        yz_flip = np.eye(3, dtype=np.float32)
        yz_flip[1, 1], yz_flip[2, 2] = -1, -1
        self.yz_flip = torch.tensor(yz_flip, device="cuda:0")

        self.renderer = renderers[mode](height, width)
        if camera_center is None:
            self.camera_center = np.array([0, 0, 0], dtype=np.float32)
        if camera_up is None:
            self.camera_up = np.array([0, 1, 0], dtype=np.float32)
        if camera_fov_y is None:
            self.camera_fov_y = 49.13434207744484 * np.pi / 180.0
        self.camera_params = None

    def forward(self, points, *args, **kwargs):

        if self.camera_params is None:
            print(
                "Camera parameters have not been set, default perspective parameters of distance = 1, elevation = 30, azimuth = 0 are being used"
            )
            self.set_look_at_parameters([0], [30], [1])

        if self.mode in [
            "VertexColorMulti",
            "VertexColorBatch",
            "TextureMulti",
            "TextureBatch",
        ]:
            assert self.camera_params[0].shape[0] == len(
                points
            ), "multi mode need the same length of camera parameters and points"
        else:
            assert (
                self.camera_params[0].shape[0] == points[0].shape[0]
            ), "Set camera parameters batch size must equal\
                batch size of passed points"

        return self.renderer(points, self.camera_params, *args, **kwargs)

    def set_look_at_parameters(self, azimuth, elevation, distance):
        from kaolin.mathutils.geometry.transformations import (
            compute_camera_params,
        )

        camera_projection_mtx = perspectiveprojectionnp(self.camera_fov_y, 1.0)
        camera_projection_mtx = torch.FloatTensor(camera_projection_mtx).cuda()

        camera_view_mtx = []
        camera_view_shift = []
        for a, e, d in zip(azimuth, elevation, distance):
            mat, pos = compute_camera_params(a, e, d)
            camera_view_mtx.append(mat)
            camera_view_shift.append(pos)
        camera_view_mtx = torch.stack(camera_view_mtx).cuda()
        camera_view_shift = torch.stack(camera_view_shift).cuda()

        self.camera_params = [
            camera_view_mtx,
            camera_view_shift,
            camera_projection_mtx,
        ]

    def set_camera_parameters(self, parameters):
        self.camera_params = parameters

    def set_camera_parameters_from_RT_K(self, Rs, ts, Ks, height, width, near=0.01, far=10.0, rot_type="mat"):
        """
        Rs: a list of rotations tensor
        ts: a list of translations tensor
        Ks: a list of camera intrinsic matrices or a single matrix
        ----
        [cam_view_R, cam_view_pos, cam_proj]
        """
        """
        aspect_ratio = width / height
        fov_x, fov_y = K_to_fov(K, height, width)
        # camera_projection_mtx = perspectiveprojectionnp(self.camera_fov_y,
        #         ratio=aspect_ratio, near=near, far=far)
        camera_projection_mtx = perspectiveprojectionnp(fov_y,
                ratio=aspect_ratio, near=near, far=far)
        """
        assert rot_type in ["mat", "quat"], rot_type
        bs = len(Rs)
        single_K = False
        if isinstance(Ks, (np.ndarray, torch.Tensor)) and Ks.ndim == 2:
            K = Ks
            camera_proj_mtx = projectiveprojection_real(K, 0, 0, width, height, near, far)
            camera_proj_mtx = torch.as_tensor(camera_proj_mtx).float().cuda()  # 4x4
            single_K = True

        camera_view_mtx = []
        camera_view_shift = []
        if not single_K:
            camera_proj_mtx = []
        for i in range(bs):
            R = Rs[i]
            t = ts[i]
            if not isinstance(R, torch.Tensor):
                R = torch.tensor(R, dtype=torch.float32, device="cuda:0")
            if not isinstance(t, torch.Tensor):
                t = torch.tensor(t, dtype=torch.float32, device="cuda:0")
            if rot_type == "quat":
                R = quat2mat_torch(R.unsqueeze(0))[0]
            cam_view_R = torch.matmul(self.yz_flip.to(R), R)
            cam_view_t = -(torch.matmul(R.t(), t))  # cam pos

            camera_view_mtx.append(cam_view_R)
            camera_view_shift.append(cam_view_t)
            if not single_K:
                K = Ks[i]
                cam_proj_mtx = projectiveprojection_real(K, 0, 0, width, height, near, far)
                cam_proj_mtx = torch.as_tensor(cam_proj_mtx).float().cuda()  # 4x4
                camera_proj_mtx.append(cam_proj_mtx)
        camera_view_mtx = torch.stack(camera_view_mtx).cuda()  # bx3x3
        camera_view_shift = torch.stack(camera_view_shift).cuda()  # bx3
        if not single_K:
            camera_proj_mtx = torch.stack(camera_proj_mtx)  # bx3x1 or bx4x4

        # print("camera view matrix: \n", camera_view_mtx, camera_view_mtx.shape) # bx3x3, camera rot?
        # print('camera view shift: \n', camera_view_shift, camera_view_shift.shape) # bx3, camera trans?
        # print('camera projection mat: \n', camera_proj_mtx, camera_proj_mtx.shape) # projection matrix, 3x1
        self.camera_params = [
            camera_view_mtx,
            camera_view_shift,
            camera_proj_mtx,
        ]
        # self.rot_type = rot_type


def K_to_fov(K, height, width):
    fx = K[0, 0]
    fy = K[1, 1]
    fov_x = 2 * np.arctan2(width, 2 * fx)  # radian
    fov_y = 2 * np.arctan2(height, 2 * fy)
    return fov_x, fov_y
