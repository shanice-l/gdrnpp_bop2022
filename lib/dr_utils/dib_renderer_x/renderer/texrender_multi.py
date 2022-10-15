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
class TexRenderMulti(nn.Module):
    def __init__(self, height, width, filtering="nearest"):
        super(TexRenderMulti, self).__init__()

        self.height = height
        self.width = width
        self.filtering = filtering

    def forward(self, points, cameras, uv_bxpx2, texture_bx3xthxtw, ts, ft_fx3=None):
        """
        points: b x [points_1xpx3, faces_fx3]
        cameras: [camera_rot_bx3x3, camera_pos_bx3, camera_proj_3x1]
        uv_bxpx2: b x [1xpx2]
        texture_bx3xthxtw: b x [1x3xthxtw]
        ts: list of translations
        ft_fx3: b x [fx3]
        """
        b = len(points)
        points3d_1xfx9_list = []
        points2d_1xfx6_list = []
        normalz_1xfx1_list = []
        normal1_1xfx3_list = []
        uv_1xfx9_list = []
        distances = np.array([t[2] for t in ts])
        dist_inds = np.argsort(distances)[::-1]  # descending order

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
        ren_ims = []
        ren_masks = []
        ren_probs = []
        for dist_ind in dist_inds:  # NOTE: not True but very close
            imfeat, improb_1xhxwx1_i = linear_rasterizer(
                self.width,
                self.height,
                points3d_1xfx9_list[dist_ind],
                points2d_1xfx6_list[dist_ind],
                normalz_1xfx1_list[dist_ind],
                uv_1xfx9_list[dist_ind],
            )
            imtexcoords = imfeat[:, :, :, :2]  # (1,H,W,2)
            hardmask = imfeat[:, :, :, 2:3]  # (1,H,W,1) mask
            # fragrement shader
            texture_1x3xthxtw = texture_bx3xthxtw[dist_ind]
            imrender_i = fragmentshader(imtexcoords, texture_1x3xthxtw, hardmask)
            ren_ims.append(imrender_i)
            ren_probs.append(improb_1xhxwx1_i)
            ren_masks.append(hardmask)

        for i in range(len(dist_inds)):
            if i == 0:
                imrender = ren_ims[0]
                improb_1xhxwx1 = ren_probs[0]
                fg_mask = ren_masks[0]
            else:
                imrender_i = ren_ims[i]
                improb_1xhxwx1_i = ren_probs[i]
                hardmask_i = ren_masks[i]
                mask_inds = torch.where(hardmask_i[0, :, :, 0] > 0.5)
                imrender[:, mask_inds[0], mask_inds[1], :] = imrender_i[:, mask_inds[0], mask_inds[1], :]
                improb_1xhxwx1[:, mask_inds[0], mask_inds[1], :] = improb_1xhxwx1_i[:, mask_inds[0], mask_inds[1], :]
                fg_mask[:, mask_inds[0], mask_inds[1], :] = hardmask_i[:, mask_inds[0], mask_inds[1], :]

        # return imrender, improb_1xhxwx1, normal1_1xfx3_list
        # TODO: we can also return instance visible masks, full masks
        return imrender, improb_1xhxwx1, normal1_1xfx3_list, fg_mask
