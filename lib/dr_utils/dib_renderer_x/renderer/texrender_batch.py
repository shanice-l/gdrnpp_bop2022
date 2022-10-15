from __future__ import print_function
from __future__ import division

from ..rasterizer import linear_rasterizer
from ..utils import datanormalize
from .fragment_shaders.frag_tex import fragmentshader
from .vertex_shaders.perpsective import perspective_projection
import torch
import torch.nn as nn
import numpy as np


##################################################################
class TexRenderBatch(nn.Module):
    def __init__(self, height, width, filtering="nearest"):
        super(TexRenderBatch, self).__init__()

        self.height = height
        self.width = width
        self.filtering = filtering

    def forward(self, points, cameras, uv_bxpx2, texture_bx3xthxtw, ft_fx3=None):
        """
        points: b x [points_1xpx3, faces_fx3]
        cameras: [camera_rot_bx3x3, camera_pos_bx3, camera_proj_3x1]
        uv_bxpx2: b x [1xpx2]
        texture_bx3xthxtw: b x [1x3xthxtw]
        ft_fx3: b x [fx3]
        """
        b = len(points)
        assert b > 0, b
        points3d_1xfx9_list = []
        points2d_1xfx6_list = []
        normalz_1xfx1_list = []
        normal1_1xfx3_list = []
        uv_1xfx9_list = []

        single_intrinsic = True
        if cameras[2].ndim == 3:
            assert cameras[2].shape[0] == b
            single_intrinsic = False

        for i in range(b):
            ##############################################################
            # first, MVP projection in vertexshader
            points_1xpx3, faces_fx3 = points[i]
            if single_intrinsic:
                cam_params = [
                    cameras[0][i : i + 1],
                    cameras[1][i : i + 1],
                    cameras[2],
                ]
            else:
                cam_params = [
                    cameras[0][i : i + 1],
                    cameras[1][i : i + 1],
                    cameras[2][i],
                ]
            # use faces_fx3 as ft_fx3 if not given
            if ft_fx3 is None:
                ft_fx3_single = faces_fx3
            else:
                ft_fx3_single = ft_fx3[i]

            (
                points3d_1xfx9,
                points2d_1xfx6,
                normal_1xfx3,
            ) = perspective_projection(points_1xpx3, faces_fx3, cam_params)

            ################################################################
            # normal

            # decide which faces are front and which faces are back
            normalz_1xfx1 = normal_1xfx3[:, :, 2:3]
            # normalz_bxfx1 = torch.abs(normalz_bxfx1)

            # normalize normal
            normal1_1xfx3 = datanormalize(normal_1xfx3, axis=2)

            ############################################################
            # second, rasterization
            uv_1xpx2 = uv_bxpx2[i]

            c0 = uv_1xpx2[:, ft_fx3_single[:, 0], :]
            c1 = uv_1xpx2[:, ft_fx3_single[:, 1], :]
            c2 = uv_1xpx2[:, ft_fx3_single[:, 2], :]
            mask = torch.ones_like(c0[:, :, :1])
            uv_1xfx9 = torch.cat((c0, mask, c1, mask, c2, mask), dim=2)

            # append data
            points3d_1xfx9_list.append(points3d_1xfx9)
            points2d_1xfx6_list.append(points2d_1xfx6)
            normalz_1xfx1_list.append(normalz_1xfx1)
            normal1_1xfx3_list.append(normal1_1xfx3)
            uv_1xfx9_list.append(uv_1xfx9)

        # put the object with larger depth earlier

        # imrender = torch.empty((1, self.height, self.width, 3), device=device, dtype=torch.float32)
        # improb_1xhxwx1 = torch.empty((1, self.height, self.width, 1), device=device, dtype=torch.float32)
        # fg_mask = torch.empty((1, self.height, self.width, 1), device=device, dtype=torch.float32)
        ren_ims = []
        ren_masks = []
        ren_probs = []
        for i in range(b):
            imfeat, improb_1xhxwx1_i = linear_rasterizer(
                self.width,
                self.height,
                points3d_1xfx9_list[i],
                points2d_1xfx6_list[i],
                normalz_1xfx1_list[i],
                uv_1xfx9_list[i],
            )
            imtexcoords = imfeat[:, :, :, :2]  # (1,H,W,2)
            hardmask = imfeat[:, :, :, 2:3]  # (1,H,W,1) mask
            # fragrement shader
            texture_1x3xthxtw = texture_bx3xthxtw[i]
            imrender_i = fragmentshader(imtexcoords, texture_1x3xthxtw, hardmask)
            ren_ims.append(imrender_i)  # 1HW3
            ren_probs.append(improb_1xhxwx1_i)
            ren_masks.append(hardmask)

        imrender = torch.cat(ren_ims, dim=0)  # bHW3
        improb_bxhxwx1 = torch.cat(ren_probs, dim=0)
        mask_bxhxwx1 = torch.cat(ren_masks, dim=0)
        # return imrender, improb_1xhxwx1, normal1_1xfx3_list
        return imrender, improb_bxhxwx1, normal1_1xfx3_list, mask_bxhxwx1
