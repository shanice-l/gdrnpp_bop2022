from __future__ import division

from ..rasterizer import linear_rasterizer
from ..utils import datanormalize
from .vertex_shaders.perpsective import perspective_projection
import torch
import torch.nn as nn

from functools import partial


def multi_apply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


##################################################################
class VCRenderBatch(nn.Module):
    """Vertex-Color Renderer Batch (batch rendering for different objects, only
    one object for each image) The original one only support batch rendering
    for a single object."""

    def __init__(self, height, width):
        super(VCRenderBatch, self).__init__()

        self.height = height
        self.width = width

    def forward(self, points, cameras, colors):
        """
        points: b x [points_1xpx3, faces_fx3]
        cameras: camera parameters
            [camera_rot_bx3x3, camera_pos_bx3, camera_proj_{b}x3x1]
        colors_list: b x [colors_1xpx3]
        """
        b = len(points)
        points3d_1xfx9_list = []
        points2d_1xfx6_list = []
        normalz_1xfx1_list = []
        normal1_1xfx3_list = []
        color_1xfx12_list = []

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
            colors_1xpx3 = colors[i]
            c0 = colors_1xpx3[:, faces_fx3[:, 0], :]
            c1 = colors_1xpx3[:, faces_fx3[:, 1], :]
            c2 = colors_1xpx3[:, faces_fx3[:, 2], :]
            mask = torch.ones_like(c0[:, :, :1])
            color_1xfx12 = torch.cat((c0, mask, c1, mask, c2, mask), dim=2)

            # append data
            points3d_1xfx9_list.append(points3d_1xfx9)
            points2d_1xfx6_list.append(points2d_1xfx6)
            normalz_1xfx1_list.append(normalz_1xfx1)
            normal1_1xfx3_list.append(normal1_1xfx3)
            color_1xfx12_list.append(color_1xfx12)

        # points3d_1xFx9 = torch.cat(points3d_1xfx9_list, dim=1)
        # points2d_1xFx6 = torch.cat(points2d_1xfx6_list, dim=1)
        # normalz_1xFx1 = torch.cat(normalz_1xfx1_list, dim=1)
        # normal1_1xFx3 = torch.cat(normal1_1xfx3_list, dim=1)
        # color_1xFx12 = torch.cat(color_1xfx12_list, dim=1)

        if True:
            imfeat_list, improb_list = multi_apply(
                linear_rasterizer,
                [self.width for _ in range(b)],
                [self.height for _ in range(b)],
                points3d_1xfx9_list,
                points2d_1xfx6_list,
                normalz_1xfx1_list,
                color_1xfx12_list,
            )
        else:  # debug
            imfeat_list, improb_list = multi_apply(
                linear_rasterizer,
                [self.width for _ in range(b)],
                [self.height for _ in range(b)],
                points3d_1xfx9_list,
                points2d_1xfx6_list,
                normalz_1xfx1_list,
                color_1xfx12_list,
                [0.02 for _ in range(b)],
                [30 for _ in range(b)],
                [1000 for _ in range(b)],
                [7000 for _ in range(b)],
                [True for _ in range(b)],
            )  # the last one is debug
        imfeat = torch.cat(imfeat_list, dim=0)  # [b,H,W,4]
        improb_bxhxwx1 = torch.cat(improb_list, dim=0)  # [b,H,W,1]
        imrender = imfeat[:, :, :, :3]  # (b,H,W,3), rgb
        hardmask = imfeat[:, :, :, 3:]  # (b,H,W,1) mask
        if False:
            import cv2

            hardmask_cpu = hardmask.detach().cpu().numpy()[0][:, :, 0]
            cv2.imshow("hardmask", hardmask_cpu)

        # return imrender, improb_1xhxwx1, normal1_1xFx3
        return imrender, improb_bxhxwx1, normal1_1xfx3_list, hardmask
