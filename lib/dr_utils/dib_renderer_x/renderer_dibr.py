import os
import os.path as osp

import numpy as np
from . import DIBRenderer
import torch
from tqdm import tqdm
import cv2

from core.utils.pose_utils import quat2mat_torch
from lib.pysixd import inout, misc
from lib.dr_utils.rep import TriangleMesh


def load_ply_models(
    obj_paths,
    texture_paths=None,
    vertex_scale=0.001,
    device="cuda",
    width=512,
    height=512,
    tex_resize=False,
):
    """
    NOTE: ignore width and height if tex_resize=False
    Args:
        vertex_scale: default 0.001 is used for bop models!
        tex_resize: resize the texture to smaller size for GPU memory saving
    Returns:
        a list of dicts
    """
    assert all([".obj" in _path for _path in obj_paths])
    models = []
    for i, obj_path in enumerate(tqdm(obj_paths)):
        model = {}
        mesh = TriangleMesh.from_obj(obj_path)
        vertices = mesh.vertices[:, :3]  # x,y,z
        colors = mesh.vertices[:, 3:6]  # rgb
        faces = mesh.faces.int()

        # normalize verts ( - center)
        vertices_max = vertices.max()
        vertices_min = vertices.min()
        vertices_middle = (vertices_max + vertices_min) / 2.0
        vertices = vertices - vertices_middle
        model["vertices"] = vertices[:, :].to(device)

        model["colors"] = colors[:, :].to(device)
        model["faces"] = faces[:, :].to(device)  # NOTE: -1

        if texture_paths is not None:
            texture = cv2.imread(texture_paths[i], cv2.IMREAD_COLOR)[:, :, ::-1].astype(np.float32) / 255.0
            if tex_resize:
                texture = cv2.resize(texture, (width, height), interpolation=cv2.INTER_AREA)
            # CHW
            texture = torch.from_numpy(texture.transpose(2, 0, 1)).to(device)

            model["face_uvs"] = mesh.uvs[:, :].to(device)
            model["face_uv_ids"] = mesh.face_textures[:, :].to(device)
            model["texture"] = texture

            # NOTE: texture_uv is None
            model["texture_uv"] = None

        models.append(model)

    return models


class Renderer_dibr(object):
    def __init__(self, height, width, mode):
        self.dib_ren = DIBRenderer(height, width, mode)

    def render_scene(
        self,
        Rs,
        ts,
        models,
        *,
        K,
        width,
        height,
        znear=0.01,
        zfar=100,
        rot_type="mat",
        with_mask=False,
        with_depth=True,
    ):
        """render a scene with m>=1 objects
        Args:
            Rs: [m,3,3] or [m,4] tensor
            ts: [m,3,] tensor
            models: list of dicts, each stores {"vertices":, "colors":, "faces":, }
            K: [3,3]
        Returns:
            a dict:
                color: (h,w,3)
                mask: (h,w) fg mask
                depth: (h,w)
        """
        ret = {}
        self.scene_ren = DIBRenderer(height, width, mode="VertexColorMulti")
        self.scene_ren.set_camera_parameters_from_RT_K(
            Rs, ts, K, height, width, near=znear, far=zfar, rot_type=rot_type
        )
        colors = [model["colors"][None] for model in models]  # m * [1, p, 3]
        points = [[model["vertices"][None], model["faces"].long()] for model in models]

        # points: list of [vertices, faces]
        # colors: list of colors
        color, im_prob, _, im_mask = self.scene_ren.forward(points=points, colors=colors)

        ret["color"] = color.squeeze()
        ret["prob"] = im_prob.squeeze()
        ret["mask"] = im_mask.squeeze()
        if with_depth:
            # transform xyz
            if not isinstance(Rs, torch.Tensor):
                Rs = torch.stack(Rs)  # list
            if rot_type == "quat":
                R_mats = quat2mat_torch(Rs)
            else:
                R_mats = Rs
            xyzs = [
                misc.transform_pts_Rt_th(model["vertices"], R_mats[_id], ts[_id])[None]
                for _id, model in enumerate(models)
            ]
            ren_xyzs, _, _, _ = self.scene_ren.forward(points=points, colors=xyzs)
            ret["depth"] = ren_xyzs[0, :, :, 2]  # bhw

        # color: hw3; mask: hw; depth: hw
        return ret

    def render_scene_tex(
        self,
        Rs,
        ts,
        models,
        *,
        K,
        width,
        height,
        znear=0.01,
        zfar=100,
        rot_type="mat",
        uv_type="vertex",
        with_mask=False,
        with_depth=True,
    ):
        """render a scene with m>=1 object for textured objects
        Args:
            Rs: [m,3,3] or [m,4] tensor
            ts: [m,3] tensor
            models: list of dict, each stores
                vertex uv: {"vertices":, "faces":, "texture":, "vertex_uvs":,}
                face uv: {"vertices":, "faces":, "texture":, "face_uvs":, "face_uv_ids":,}
            K: [3,3]
            uv_type: `vertex` | `face`
        Returns:
            dict:
                color: (h,w,3)
                mask: (h,w) fg mask (to get instance masks, use batch mode)
                depth: (h,w)
        """
        ret = {}
        self.scene_ren = DIBRenderer(height, width, mode="TextureMulti")
        self.scene_ren.set_camera_parameters_from_RT_K(
            Rs, ts, K, height, width, near=znear, far=zfar, rot_type=rot_type
        )
        # points: list of [vertices, faces]
        points = [[model["vertices"][None], model["faces"].long()] for model in models]
        if uv_type == "vertex":
            uv_bxpx2 = [model["vertex_uvs"][None] for model in models]
        else:  # face uv
            uv_bxpx2 = [model["face_uvs"][None] for model in models]
            ft_fx3_list = [model["face_uv_ids"] for model in models]
        texture_bx3xthxtw = [model["texture"][None] for model in models]

        dib_ren_im, dib_ren_prob, _, dib_ren_mask = self.scene_ren.forward(
            points=points,
            uv_bxpx2=uv_bxpx2,
            texture_bx3xthxtw=texture_bx3xthxtw,
            ts=ts,
            ft_fx3=ft_fx3_list,
        )

        ret["color"] = dib_ren_im.squeeze()
        ret["prob"] = dib_ren_prob.squeeze()
        ret["mask"] = dib_ren_mask.squeeze()

        if with_depth:
            # transform xyz
            # NOTE: check whether it should be in [0, 1] (maybe need to record min, max and denormalize later)
            if not isinstance(Rs, torch.Tensor):
                Rs = torch.stack(Rs)  # list
            if rot_type == "quat":
                R_mats = quat2mat_torch(Rs)
            else:
                R_mats = Rs
            xyzs = [
                misc.transform_pts_Rt_th(model["vertices"], R_mats[_id], ts[_id])[None]
                for _id, model in enumerate(models)
            ]
            dib_ren_vc_batch = DIBRenderer(height, width, mode="VertexColorMulti")
            dib_ren_vc_batch.set_camera_parameters(self.scene_ren.camera_params)
            ren_xyzs, _, _, _ = dib_ren_vc_batch.forward(points=points, colors=xyzs)
            ret["depth"] = ren_xyzs[0, :, :, 2]  # hw

        # color: hw3; mask: hw; depth: hw
        return ret

    def render_batch(
        self,
        Rs,
        ts,
        models,
        *,
        Ks,
        width,
        height,
        znear=0.01,
        zfar=100,
        rot_type="mat",
        mode=["color", "depth"],
    ):
        """render a batch (vertex color), each contain one object
        Args:
            Rs (tensor): [b,3,3] or [b,4]
            ts (tensor): [b,3,]
            models (list of dicts): each stores {"vertices":, "colors":, "faces":, }
            Ks (tensor): [b,3,3]
            mode: color, depth, mask, xyz (one or more must be given)
        Returns:
            dict:
                color: bhw3
                mask: bhw
                depth: bhw
                xyz: bhw3
                probs: bhw
        """
        assert self.dib_ren.mode in ["VertexColorBatch"], self.dib_ren.mode
        ret = {}
        self.dib_ren.set_camera_parameters_from_RT_K(Rs, ts, Ks, height, width, near=znear, far=zfar, rot_type=rot_type)

        colors = [model["colors"][None] for model in models]  # b x [1, p, 3]
        points = [[model["vertices"][None], model["faces"].long()] for model in models]

        # points: list of [vertices, faces]
        # colors: list of colors
        color, im_prob, _, im_mask = self.dib_ren.forward(points=points, colors=colors)
        ret["color"] = color
        ret["prob"] = im_prob.squeeze(-1)
        ret["mask"] = im_mask.squeeze(-1)

        if "depth" in mode:
            # transform xyz
            if not isinstance(Rs, torch.Tensor):
                Rs = torch.stack(Rs)  # list
            if rot_type == "quat":
                R_mats = quat2mat_torch(Rs)
            else:
                R_mats = Rs
            xyzs = [
                misc.transform_pts_Rt_th(model["vertices"], R_mats[_id], ts[_id])[None]
                for _id, model in enumerate(models)
            ]
            ren_xyzs, _, _, _ = self.dib_ren.forward(points=points, colors=xyzs)
            ret["depth"] = ren_xyzs[:, :, :, 2]  # bhw

        if "xyz" in mode:  # TODO: check this
            obj_xyzs = [model["vertices"][None] for _id, model in enumerate(models)]
            ren_obj_xyzs, _, _, _ = self.dib_ren.forward(points=points, colors=obj_xyzs)
            ret["xyz"] = ren_obj_xyzs
        return ret

    def render_batch_tex(
        self,
        Rs,
        ts,
        models,
        *,
        Ks,
        width,
        height,
        znear=0.01,
        zfar=100,
        uv_type="vertex",
        rot_type="mat",
        mode=["color", "depth"],
    ):
        """render a batch for textured objects
        Args:
            Rs: [b,3,3] or [b,4] tensor
            ts: [b,3] tensor
            models: list of dict, each stores
                vertex uv: {"vertices":, "faces":, "texture":, "vertex_uvs":,}
                face uv: {"vertices":, "faces":, "texture":, "face_uvs":, "face_uv_ids":,}
            Ks: [b,3,3] or [3,3]
            uv_type: `vertex` | `face`
            mode: color, depth, mask, xyz (one or more must be given)
        Returns:
            dict:
                color: bhw3
                mask: bhw
                depth: bhw
                xyz: bhw3
        """
        assert self.dib_ren.mode in ["TextureBatch"], self.dib_ren.mode
        ret = {}
        self.dib_ren.set_camera_parameters_from_RT_K(Rs, ts, Ks, height, width, near=znear, far=zfar, rot_type=rot_type)
        # points: list of [vertices, faces]
        points = [[model["vertices"][None], model["faces"].long()] for model in models]
        if uv_type == "vertex":
            uv_bxpx2 = [model["vertex_uvs"][None] for model in models]
        else:  # face uv
            uv_bxpx2 = [model["face_uvs"][None] for model in models]
            ft_fx3_list = [model["face_uv_ids"] for model in models]
        texture_bx3xthxtw = [model["texture"][None] for model in models]

        # points: list of [vertices, faces]
        # colors: list of colors
        dib_ren_im, dib_ren_prob, _, dib_ren_mask = self.dib_ren.forward(
            points=points,
            uv_bxpx2=uv_bxpx2,
            texture_bx3xthxtw=texture_bx3xthxtw,
            ft_fx3=ft_fx3_list,
        )

        ret["color"] = dib_ren_im
        ret["prob"] = dib_ren_prob.squeeze(-1)  # bhw1 -> bhw
        ret["mask"] = dib_ren_mask.squeeze(-1)  # bhw1 -> bhw

        if "depth" in mode:
            # transform xyz
            # NOTE: check whether it should be in [0, 1] (maybe need to record min, max and denormalize later)
            if not isinstance(Rs, torch.Tensor):
                Rs = torch.stack(Rs)  # list
            if rot_type == "quat":
                R_mats = quat2mat_torch(Rs)
            else:
                R_mats = Rs
            xyzs = [
                misc.transform_pts_Rt_th(model["vertices"], R_mats[_id], ts[_id])[None]
                for _id, model in enumerate(models)
            ]
            dib_ren_vc_batch = DIBRenderer(height, width, mode="VertexColorBatch")
            dib_ren_vc_batch.set_camera_parameters(self.dib_ren.camera_params)
            ren_xyzs, _, _, _ = dib_ren_vc_batch.forward(points=points, colors=xyzs)
            if "depth" in mode:
                ret["depth"] = ren_xyzs[:, :, :, 2]  # bhw

        if "xyz" in mode:  # TODO: check this
            obj_xyzs = [model["vertices"][None] for _id, model in enumerate(models)]
            dib_ren_vc_batch = DIBRenderer(height, width, mode="VertexColorBatch")
            dib_ren_vc_batch.set_camera_parameters(self.dib_ren.camera_params)
            ren_obj_xyzs, _, _, _ = dib_ren_vc_batch.forward(points=points, colors=obj_xyzs)
            ret["xyz"] = ren_obj_xyzs
        return ret  # bxhxwx3 rgb, bhw prob/mask/depth
