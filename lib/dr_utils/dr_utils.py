# differentiable renderer utils
import cv2
import numpy as np
import torch
import os.path as osp
from tqdm import tqdm
import hashlib
import logging
import mmcv
from .rep import TriangleMesh
from .dib_renderer_x import DIBRenderer
from core.utils.pose_utils import quat2mat_torch
from lib.utils.utils import iprint
from lib.pysixd import misc


def load_objs(
    obj_paths,
    texture_paths=None,
    height=480,
    width=640,
    tex_resize=True,
    tex_fmt="CHW",
    tex_vflip=False,
):
    """
    NOTE: ignore width, height if tex_resize=False
    """
    assert all([".obj" in _path for _path in obj_paths])

    if texture_paths is not None:
        assert len(obj_paths) == len(texture_paths)
    models = []
    for i, obj_path in enumerate(tqdm(obj_paths)):
        model = {}
        mesh = TriangleMesh.from_obj(obj_path)
        vertices = mesh.vertices[:, :3]  # x,y,z
        colors = mesh.vertices[:, 3:6]  # rgb
        faces = mesh.faces.int()
        ###########################
        # normalize verts ( - center)
        ###########################
        vertices_max = vertices.max()
        vertices_min = vertices.min()
        vertices_middle = (vertices_max + vertices_min) / 2.0
        vertices = vertices - vertices_middle
        model["vertices"] = vertices[None, :, :].cuda()
        model["colors"] = colors[None, :, :].cuda()
        model["faces"] = faces[None, :, :].cuda()  # NOTE: -1
        if texture_paths is not None:
            uvs = mesh.uvs
            face_textures = mesh.face_textures  # NOTE: -1
            assert osp.exists(texture_paths[i]), texture_paths[i]
            if tex_vflip:
                texture = cv2.imread(texture_paths[i], cv2.IMREAD_COLOR)[::-1, :, ::-1].astype(np.float32) / 255.0
            else:
                texture = cv2.imread(texture_paths[i], cv2.IMREAD_COLOR)[:, :, ::-1].astype(np.float32) / 255.0
            if tex_resize:
                texture = cv2.resize(texture, (width, height), interpolation=cv2.INTER_AREA)
            # print('texture map: ', texture.shape)
            if tex_fmt == "CHW":
                texture = torch.from_numpy(texture.transpose(2, 0, 1)[None, :, :, :]).cuda()
            else:  # HWC
                texture = torch.from_numpy(texture[None, :, :, :]).cuda()
            model["face_uvs"] = uvs[None, :, :].cuda()
            model["face_uv_ids"] = face_textures[None, :, :].cuda()
            model["texture"] = texture.cuda()
        models.append(model)

    return models


def render_dib_vc_batch(
    ren,
    Rs,
    ts,
    Ks,
    obj_ids,
    models,
    rot_type="quat",
    H=480,
    W=640,
    near=0.01,
    far=100.0,
    with_depth=False,
):
    """
    Args:
        ren: A DIB-renderer
        models: All models loaded by load_objs
    """
    assert ren.mode in ["VertexColorBatch"], ren.mode
    bs = len(Rs)
    if len(Ks) == 1:
        Ks = [Ks[0] for _ in range(bs)]
    ren.set_camera_parameters_from_RT_K(Rs, ts, Ks, height=H, width=W, near=near, far=far, rot_type=rot_type)
    colors = [models[_id]["colors"] for _id in obj_ids]  # b x [1, p, 3]
    points = [[models[_id]["vertices"], models[_id]["faces"][0].long()] for _id in obj_ids]

    # points: list of [vertices, faces]
    # colors: list of colors
    predictions, im_probs, _, im_masks = ren.forward(points=points, colors=colors)
    if with_depth:
        # transform xyz
        if not isinstance(Rs, torch.Tensor):
            Rs = torch.stack(Rs)  # list
        if rot_type == "quat":
            R_mats = quat2mat_torch(Rs)
        else:
            R_mats = Rs
        xyzs = [
            misc.transform_pts_Rt_th(models[obj_id]["vertices"][0], R_mats[_id], ts[_id])[None]
            for _id, obj_id in enumerate(obj_ids)
        ]
        ren_xyzs, _, _, _ = ren.forward(points=points, colors=xyzs)
        depth = ren_xyzs[:, :, :, 2]  # bhw
    else:
        depth = None
    # bxhxwx3 rgb, bhw1 prob, bhw1 mask, bhw depth
    return predictions, im_probs, im_masks, depth


def render_dib_tex_batch(
    ren,
    Rs,
    ts,
    Ks,
    obj_ids,
    models,
    rot_type="quat",
    H=480,
    W=640,
    near=0.01,
    far=100.0,
    with_depth=False,
):
    assert ren.mode in ["TextureBatch"], ren.mode
    bs = len(Rs)
    if len(Ks) == 1:
        Ks = [Ks[0] for _ in range(bs)]
    ren.set_camera_parameters_from_RT_K(Rs, ts, Ks, height=H, width=W, near=near, far=far, rot_type=rot_type)
    # points: list of [vertices, faces]
    points = [[models[_id]["vertices"], models[_id]["faces"][0].long()] for _id in obj_ids]
    uv_bxpx2 = [models[_id]["face_uvs"] for _id in obj_ids]
    texture_bx3xthxtw = [models[_id]["texture"] for _id in obj_ids]
    ft_fx3_list = [models[_id]["face_uv_ids"][0] for _id in obj_ids]

    # points: list of [vertices, faces]
    # colors: list of colors
    dib_ren_im, dib_ren_prob, _, dib_ren_mask = ren.forward(
        points=points,
        uv_bxpx2=uv_bxpx2,
        texture_bx3xthxtw=texture_bx3xthxtw,
        ft_fx3=ft_fx3_list,
    )

    if with_depth:
        # transform xyz
        if not isinstance(Rs, torch.Tensor):
            Rs = torch.stack(Rs)  # list
        if rot_type == "quat":
            R_mats = quat2mat_torch(Rs)
        else:
            R_mats = Rs
        xyzs = [
            misc.transform_pts_Rt_th(models[obj_id]["vertices"][0], R_mats[_id], ts[_id])[None]
            for _id, obj_id in enumerate(obj_ids)
        ]
        dib_ren_vc_batch = DIBRenderer(height=H, width=W, mode="VertexColorBatch")
        dib_ren_vc_batch.set_camera_parameters(ren.camera_params)
        ren_xyzs, _, _, _ = dib_ren_vc_batch.forward(points=points, colors=xyzs)
        depth = ren_xyzs[:, :, :, 2]  # bhw
    else:
        depth = None
    return (
        dib_ren_im,
        dib_ren_prob,
        dib_ren_mask,
        depth,
    )  # bxhxwx3 rgb, bhw1 prob/mask, bhw depth


def render_dib_vc_multi(
    ren,
    Rs,
    ts,
    K,
    obj_ids,
    models,
    rot_type="quat",
    H=480,
    W=640,
    near=0.01,
    far=100.0,
):
    assert ren.mode in ["VertexColorMulti"], ren.mode
    ren.set_camera_parameters_from_RT_K(Rs, ts, K, height=H, width=W, near=near, far=far, rot_type=rot_type)
    colors = [models[_id]["colors"] for _id in obj_ids]  # b x [1, p, 3]
    points = [[models[_id]["vertices"], models[_id]["faces"][0].long()] for _id in obj_ids]

    # points: list of [vertices, faces]
    # colors: list of colors
    predictions, im_prob, _, im_mask = ren.forward(points=points, colors=colors)
    # TODO: add depth
    return predictions, im_prob, im_mask  # 1xhxwx3 rgb


def render_dib_tex_multi(
    ren,
    Rs,
    ts,
    K,
    obj_ids,
    models,
    rot_type="quat",
    H=480,
    W=640,
    near=0.01,
    far=100.0,
):
    assert ren.mode in ["TextureMulti"], ren.mode
    ren.set_camera_parameters_from_RT_K(Rs, ts, K, height=H, width=W, near=near, far=far, rot_type=rot_type)
    # points: list of [vertices, faces]
    points = [[models[_id]["vertices"], models[_id]["faces"][0].long()] for _id in obj_ids]
    uv_bxpx2 = [models[_id]["face_uvs"] for _id in obj_ids]
    texture_bx3xthxtw = [models[_id]["texture"] for _id in obj_ids]
    ft_fx3_list = [models[_id]["face_uv_ids"][0] for _id in obj_ids]

    dib_ren_im, dib_ren_prob, _, dib_ren_mask = ren.forward(
        points=points,
        uv_bxpx2=uv_bxpx2,
        texture_bx3xthxtw=texture_bx3xthxtw,
        ts=ts,
        ft_fx3=ft_fx3_list,
    )
    # TODO: add depth
    return (
        dib_ren_im,
        dib_ren_prob,
        dib_ren_mask,
    )  # 1xhxwx3 rgb, (1,h,w,1) prob/mask
