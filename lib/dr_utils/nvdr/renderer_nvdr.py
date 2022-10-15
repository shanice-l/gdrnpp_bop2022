import os

import cv2
import numpy as np
import nvdiffrast.torch as dr
import torch
from pytorch3d.structures import Meshes, list_to_packed, list_to_padded
from tqdm import tqdm

from core.utils.pose_utils import quat2mat_torch
from lib.pysixd import inout, misc


_GPU_ID = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])


def _list_to_packed(x):
    """return packed tensor and ranges."""
    (
        x_packed,
        num_items,
        item_packed_first_idx,
        item_packed_to_list_idx,
    ) = list_to_packed(x)
    ranges = torch.stack([item_packed_first_idx, num_items], dim=1).to(dtype=torch.int32, device="cpu")
    return x_packed, ranges


def _get_color_depth_xyz_code(mode):
    all_modes = ["color", "depth", "xyz"]
    return [int(_m in mode) for _m in all_modes]


def _get_depth_xyz_code(mode):
    all_modes = ["depth", "xyz"]
    return [int(_m in mode) for _m in all_modes]


def load_ply_models(
    model_paths,
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
    ply_models = [inout.load_ply(model_path, vertex_scale=vertex_scale) for model_path in model_paths]
    models = []
    for i, ply_model in enumerate(tqdm(ply_models)):
        vertices = ply_model["pts"]
        faces = ply_model["faces"]
        if "colors" in ply_model:
            colors = ply_model["colors"]
            if colors.max() > 1.1:
                colors = colors / 255.0
        else:
            colors = np.zeros_like(vertices)
            colors[:, 0] = 223.0 / 255
            colors[:, 1] = 214.0 / 255
            colors[:, 2] = 205.0 / 255

        if texture_paths is not None:
            texture_path = texture_paths[i]
            texture = cv2.imread(texture_path, cv2.IMREAD_COLOR)[::-1, :, ::-1].astype(np.float32) / 255.0
            if tex_resize:
                texture = cv2.resize(texture, (width, height), interpolation=cv2.INTER_AREA)
            # print('texture map: ', texture.shape)
            texture = torch.tensor(texture, device=device, dtype=torch.float32)
            # uv coordinates for vertices
            texture_uv = torch.tensor(ply_model["texture_uv"].astype("float32"), device=device)
        else:
            texture = None
            texture_uv = None

        vertices = torch.tensor(np.ascontiguousarray(vertices.astype("float32")), device=device)
        # for ply, already 0-based
        faces = torch.tensor(faces.astype("int32"), device=device)
        colors = torch.tensor(colors.astype("float32"), device=device)

        models.append(
            {
                "vertices": vertices,
                "faces": faces,
                "colors": colors,
                "texture": texture,
                "vertex_uvs": texture_uv,
            }
        )

    return models


class Renderer_nvdr(object):
    def __init__(self, output_db=True, glctx_mode="manual", device="cuda", gpu_id=None):
        """output_db (bool): Compute and output image-space derivates of
        barycentrics.

        glctx_mode: OpenGL context handling mode. Valid values are 'manual' and 'automatic'.
        """
        if glctx_mode == "auto":
            glctx_mode = "automatic"
        assert glctx_mode in ["automatic", "manual"], glctx_mode
        self._glctx_mode = glctx_mode

        self.output_db = output_db
        self._diff_attrs = "all" if output_db else None
        self._glctx = dr.RasterizeGLContext(output_db=output_db, mode=glctx_mode, device=gpu_id)
        if glctx_mode == "manual":
            self._glctx.set_context()
        self._device = device

        self._V = self._model_view()  # view matrix, (I_4x4)

    def _transform_pos(self, mtx, pos):
        """# Transform vertex positions to clip space
        Args:
            mtx: transform matrix [4, 4]
            pos: [n,3] vertices
        Returns:
            [1,n,4]
        """
        assert pos.shape[1] == 3, pos.shape
        t_mtx = torch.from_numpy(mtx).to(device=self._device) if isinstance(mtx, np.ndarray) else mtx
        # (x,y,z) -> (x,y,z,1)
        posw = torch.cat([pos, torch.ones([pos.shape[0], 1]).to(device=self._device)], dim=1)
        # (n,4)x(4, 4)-->(1,n,4)
        return torch.matmul(posw, t_mtx.t())[None, ...]

    def _transform_pos_batch(self, mtx, pos):
        """# Transform vertex positions to clip space
        Args:
            mtx: transform matrix [B, 4, 4]
            pos: [B,n,3] vertices
        Returns:
            [B,n,4]
        """
        bs = mtx.shape[0]
        assert pos.ndim == 3 and pos.shape[-1] == 3, pos.shape
        t_mtx = torch.from_numpy(mtx).to(device=self._device) if isinstance(mtx, np.ndarray) else mtx
        # (x,y,z) -> (x,y,z,1)
        num_pts = pos.shape[1]
        _ones = torch.ones([bs, num_pts, 1]).to(device=self._device, dtype=pos.dtype)
        posw = torch.cat([pos, _ones], dim=-1)  # [B,n,4]
        # (B,n,4)x(B,4,4)-->(B,n,4)
        return torch.matmul(posw, t_mtx.t())

    def _get_poses(self, Rs, ts, rot_type="mat"):
        assert rot_type in ["mat", "quat"], rot_type
        if rot_type == "quat":
            rots = quat2mat_torch(Rs)
        else:
            rots = Rs
        num = rots.shape[0]
        assert ts.shape[0] == num, ts.shape
        dtype = rots.dtype
        poses = torch.cat([rots, ts.view(num, 3, 1)], dim=2)  # [num_objs,3,4]
        poses_4x4 = torch.eye(4).repeat(num, 1, 1).to(poses)
        poses_4x4[:, :3, :] = poses
        return poses_4x4.to(device=self._device, dtype=dtype)

    def _model_view(self):
        V = np.eye(4)
        V = np.ascontiguousarray(V, np.float32)
        return torch.tensor(V, device=self._device)

    def _projection(self, x=0.1, n=0.01, f=50.0):
        P = np.array(
            [
                [n / x, 0, 0, 0],
                [0, n / -x, 0, 0],
                [0, 0, -(f + n) / (f - n), -(2 * f * n) / (f - n)],
                [0, 0, -1, 0],
            ]
        ).astype(np.float32)
        return torch.tensor(np.ascontiguousarray(P, np.float32), device=self._device)

    def _projection_real(self, cam, x0, y0, w, h, nc=0.01, fc=10.0):
        # this is for center view
        # NOTE: only return a 3x1 vector (diagonal??)
        q = -(fc + nc) / float(fc - nc)
        qn = -2 * (fc * nc) / float(fc - nc)
        fx = cam[0, 0]
        fy = cam[1, 1]
        px = cam[0, 2]
        # HACK: lm: -4, ycbv: -2.5
        py = cam[1, 2]  # + self.v_offset
        """
        # transpose: compensate for the flipped image
        proj_T = [
                [2*fx/w,          0,                0,  0],
                [0,               2*fy/h,           0,  0],
                [(-2*px+w+2*x0)/w, (2*py-h+2*y0)/h, q,  -1],
                [0,               0,                qn, 0],
            ]
            sometimes: proj_T[1,:] *= -1, proj_T[2,:] *= -1
            # Third column is standard glPerspective and sets near and far planes
        """
        # Draw our images upside down, so that all the pixel-based coordinate systems are the same
        if isinstance(cam, np.ndarray):
            proj_T = np.zeros((4, 4), dtype=np.float32)
        elif isinstance(cam, torch.Tensor):
            proj_T = torch.zeros(4, 4).to(cam)
        else:
            raise TypeError("cam should be ndarray or tensor, got {}".format(type(cam)))
        proj_T[0, 0] = 2 * fx / w
        proj_T[1, 0] = -2 * cam[0, 1] / w  # =0
        proj_T[1, 1] = 2 * fy / h
        proj_T[2, 0] = (-2 * px + w + 2 * x0) / w
        proj_T[2, 1] = (+2 * py - h + 2 * y0) / h
        proj_T[2, 2] = q
        proj_T[3, 2] = qn
        proj_T[2, 3] = -1.0

        proj_T[2, :] *= -1
        if isinstance(cam, np.ndarray):
            P = proj_T.T
            return torch.tensor(np.ascontiguousarray(P, np.float32), device=self._device)
        elif isinstance(cam, torch.Tensor):
            P = proj_T.t()
            return P.contiguous().to(device=self._device)
        else:
            raise TypeError("cam should be ndarray or tensor, got {}".format(type(cam)))

    def close(self):
        if self._glctx_mode == "manual":
            self._glctx.release_context()

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
        antialias=True,
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
        P = self._projection_real(cam=K, x0=0, y0=0, w=width, h=height, nc=znear, fc=zfar)
        # Modelview + projection matrix.
        mvp = torch.matmul(P, self._V)  # [4,4]
        assert rot_type in ["mat", "quat"], f"Unknown rot_type: {rot_type}"
        poses_4x4 = self._get_poses(Rs, ts, rot_type=rot_type)  # [m,4,4]
        mtx = torch.matmul(mvp.view(1, 4, 4), poses_4x4)  # [m,4,4]

        vertices_list = [torch.squeeze(model["vertices"]) for model in models]
        nvert_list = [_v.shape[0] for _v in vertices_list]
        vert_offset_list = [0] + np.cumsum(nvert_list).tolist()[:-1]

        model_colors_list = [torch.squeeze(model["colors"]) for model in models]
        if with_depth:
            pc_cam_list = [misc.transform_pts_Rt_th(vertices, R, t) for vertices, R, t in zip(vertices_list, Rs, ts)]
            colors_depths_list = [
                torch.cat([model_colors, pc_cam[:, 2:3]], dim=1)
                for model_colors, pc_cam in zip(model_colors_list, pc_cam_list)
            ]
            colors_depths_all = torch.cat(colors_depths_list, dim=0)
        else:
            # no depth
            colors_depths_all = torch.cat(model_colors_list, dim=0)

        ####### render ###############
        # list of [1, n, 4]
        pos_clip_list = [self._transform_pos(mtx_i, vertices) for mtx_i, vertices in zip(mtx, vertices_list)]
        pos_clip_all = torch.cat(pos_clip_list, dim=1)

        pos_idx_list = [torch.squeeze(model["faces"].to(torch.int32)) for model in models]
        pos_idx_list = [(pos_idx + _offset).to(torch.int32) for pos_idx, _offset in zip(pos_idx_list, vert_offset_list)]
        pos_idx_all = torch.cat(pos_idx_list, dim=0)

        rast_out, _ = dr.rasterize(self._glctx, pos_clip_all, pos_idx_all, resolution=[height, width])
        color_depth, _ = dr.interpolate(colors_depths_all[None, ...], rast_out, pos_idx_all)
        if antialias:
            color_depth = dr.antialias(color_depth, rast_out, pos_clip_all, pos_idx_all)

        color = color_depth[0, :, :, :3]
        ret = {"color": color}
        if with_mask:
            mask = torch.clamp(rast_out[0, :, :, -1], 0, 1)
            ret["mask"] = mask
        if with_depth:
            depth = color_depth[0, :, :, 3]
            ret["depth"] = depth
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
        enable_mip=True,
        max_mip_level=10,
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
        P = self._projection_real(cam=K, x0=0, y0=0, w=width, h=height, nc=znear, fc=zfar)
        # Modelview + projection matrix.
        mvp = torch.matmul(P, self._V)
        assert rot_type in ["mat", "quat"], f"Unknown rot_type: {rot_type}"
        poses_4x4 = self._get_poses(Rs, ts, rot_type=rot_type)  # [m,4,4]
        mtx = torch.matmul(mvp, poses_4x4)  # [m,4,4]

        verts_list = [torch.squeeze(model["vertices"]) for model in models]
        faces_list = [torch.squeeze(model["faces"].to(torch.int32)) for model in models]
        meshes = Meshes(verts=verts_list, faces=faces_list)
        # verts_packed = meshes.verts_packed()  # [sum(Vi),3]
        faces_packed = meshes.faces_packed().to(dtype=torch.int32)  # [sum(Fi),3]
        faces_ranges = torch.stack(
            [
                meshes.mesh_to_faces_packed_first_idx(),
                meshes.num_faces_per_mesh(),
            ],
            dim=1,
        ).to(dtype=torch.int32, device="cpu")

        if with_depth:
            pc_cam_list = [misc.transform_pts_Rt_th(_v, R, t) for _v, R, t in zip(verts_list, Rs, ts)]
            pc_cam_packed, _ = _list_to_packed(pc_cam_list)  # [sum(Vi),3]

        ####### render ###############
        # list of [n, 4]
        pos_clip_list = [self._transform_pos(mtx_i, _v)[0] for mtx_i, _v in zip(mtx, verts_list)]
        pos_clip_packed, _ = _list_to_packed(pos_clip_list)

        assert uv_type in ["vertex", "face"], uv_type
        if uv_type == "vertex":
            uv_list = [torch.squeeze(model["vertex_uvs"]) for model in models]
            uv_packed, _ = _list_to_packed(uv_list)
            uv_idx_packed = faces_packed  # faces
        else:  # face uv
            uv_list = [torch.squeeze(model["face_uvs"]) for model in models]
            uv_packed, _ = _list_to_packed(uv_list)
            uv_idx_list = [torch.squeeze(model["face_uv_ids"]).to(dtype=torch.int32) for model in models]
            uv_idx_packed, uv_idx_ranges = _list_to_packed(uv_idx_list)
        # NOTE: must be the same size
        tex_list = [torch.squeeze(model["texture"]) for model in models]
        tex_batch = torch.stack(tex_list, dim=0)  # [m,H,W,3]

        # Render as a batch first -----------------------------------------------------
        rast_out, rast_out_db = dr.rasterize(
            self._glctx,
            pos_clip_packed,
            faces_packed,
            ranges=faces_ranges,
            resolution=[height, width],
        )
        if enable_mip:
            texc, texd = dr.interpolate(
                uv_packed,
                rast_out,
                uv_idx_packed,
                rast_db=rast_out_db,
                diff_attrs=self._diff_attrs,
            )
            color = dr.texture(
                tex_batch,
                texc,
                texd,
                filter_mode="linear-mipmap-linear",
                max_mip_level=max_mip_level,
            )
        else:
            texc, _ = dr.interpolate(uv_packed, rast_out, uv_idx_packed)
            color = dr.texture(tex_batch, texc, filter_mode="linear")

        masks = torch.clamp(rast_out[:, :, :, -1:], 0, 1)  # bhw1
        color = color * masks  # Mask out background.
        if with_depth:
            im_pc_cam, _ = dr.interpolate(pc_cam_packed, rast_out, faces_packed)
            depth = im_pc_cam[..., 2:3] * masks  # Mask out background.
        else:
            depth = None
        # use the batched results as a scene --------------------------------------------------
        ret = self._batch_to_scene_color_depth(
            color,
            ts=ts,
            depths=depth,
            masks=masks,
            with_mask=with_mask,
            with_depth=with_depth,
        )
        # color: hw3; mask: hw; depth: hw
        return ret

    def _batch_to_scene_color_depth(
        self,
        colors,
        ts,
        depths=None,
        masks=None,
        with_mask=False,
        with_depth=True,
    ):
        """
        Args:
            colors: [b,h,w,3]
            depths: [b,h,w,1]
            ts: [b,3]
            masks: [b,h,w,1] or None
        Returns:
            dict:
                scene_color: hw3
                scene_mask: hw
                scene_depth: hw
        """
        tz_list = [_t[2] for _t in ts]  # render farther object first
        dist_inds = np.argsort(tz_list)[::-1]  # descending order
        if masks is None:
            assert depths is not None
            masks = (depths > 0).to(torch.float32)
        for i, dist_i in enumerate(dist_inds):
            if i == 0:
                scene_color = colors[dist_i]
                if with_mask:
                    scene_mask = masks[dist_i, :, :, 0]
                if with_depth:
                    scene_depth = depths[dist_i, :, :, 0]
            else:
                cur_mask = torch.clamp(masks[dist_i], 0, 1)
                mask_inds = torch.where(cur_mask[:, :, 0] > 0.5)
                scene_color[mask_inds[0], mask_inds[1], :] = colors[dist_i, mask_inds[0], mask_inds[1], :]
                if with_mask:
                    scene_mask[mask_inds[0], mask_inds[1]] = masks[dist_i, mask_inds[0], mask_inds[1], 0]
                if with_depth:
                    scene_depth[mask_inds[0], mask_inds[1]] = depths[dist_i, mask_inds[0], mask_inds[1], 0]
        ret = {"color": scene_color}
        if with_mask:
            ret["mask"] = scene_mask
        if with_depth:
            ret["depth"] = scene_depth
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
        antialias=True,
        mode=["color", "depth"],
    ):
        """render a batch (vertex color), each contain one object
        Args:
            Rs (tensor): [b,3,3] or [b,4]
            ts (tensor): [b,3,]
            models (list of dicts): each stores {"vertices":, "colors":, "faces":, }
            Ks (ndarray): [b,3,3] or [3,3]
            mode: color, depth, mask, xyz (one or more must be given)
        Returns:
            dict:
                color: bhw3
                mask: bhw
                depth: bhw
                xyz: bhw3
        """
        assert len(mode) >= 1, mode
        bs = Rs.shape[0]
        if not isinstance(Ks, (tuple, list)) and Ks.ndim == 2:
            if isinstance(Ks, torch.Tensor):
                Ks = [Ks.clone() for _ in range(bs)]
            elif isinstance(Ks, np.ndarray):
                Ks = [Ks.copy() for _ in range(bs)]
            else:
                raise TypeError(f"Unknown type of Ks: {type(Ks)}")

        Ps = [self._projection_real(cam=K, x0=0, y0=0, w=width, h=height, nc=znear, fc=zfar) for K in Ks]
        Ps = torch.stack(Ps, dim=0)  # [b,4,4]
        # Modelview + projection matrix.
        mvp = torch.matmul(Ps, self._V)  # [b,4,4]
        assert rot_type in ["mat", "quat"], f"Unknown rot_type: {rot_type}"
        poses_4x4 = self._get_poses(Rs, ts, rot_type=rot_type)  # [b,4,4]
        mtx = torch.matmul(mvp, poses_4x4)  # [b,4,4]

        verts_list = [torch.squeeze(model["vertices"]) for model in models]
        faces_list = [torch.squeeze(model["faces"].to(torch.int32)) for model in models]
        meshes = Meshes(verts=verts_list, faces=faces_list)
        # verts_packed = meshes.verts_packed()  # [sum(Vi),3]
        faces_packed = meshes.faces_packed().to(dtype=torch.int32)  # [sum(Fi),3]
        faces_ranges = torch.stack(
            [
                meshes.mesh_to_faces_packed_first_idx(),
                meshes.num_faces_per_mesh(),
            ],
            dim=1,
        ).to(dtype=torch.int32, device="cpu")

        ####### render the batch --------------------
        # list of [Vi, 4]
        pos_clip_list = [self._transform_pos(mtx_i, _v)[0] for mtx_i, _v in zip(mtx, verts_list)]
        pos_clip_packed, _ = _list_to_packed(pos_clip_list)  # [sum(Vi),4]

        rast_out, _ = dr.rasterize(
            self._glctx,
            pos_clip_packed,
            faces_packed,
            ranges=faces_ranges,
            resolution=[height, width],
        )
        ret = {}
        if "mask" in mode:
            mask = torch.clamp(rast_out[:, :, :, -1], 0, 1)
            ret["mask"] = mask

        color_depth_xyz_code = _get_color_depth_xyz_code(mode)  # color, depth, xyz
        if sum(color_depth_xyz_code) > 0:
            # color, depth, xyz
            if "color" in mode:
                model_colors_list = [torch.squeeze(model["colors"]) for model in models]
            else:
                model_colors_list = None
            if "depth" in mode:
                pc_cam_list = [misc.transform_pts_Rt_th(verts, R, t) for verts, R, t in zip(verts_list, Rs, ts)]
            else:
                pc_cam_list = None
            colors_depths_verts_list = []
            for mesh_i, verts in enumerate(verts_list):
                colors_depths_verts_i = []
                if "color" in mode:
                    colors_depths_verts_i.append(model_colors_list[mesh_i])
                if "depth" in mode:
                    colors_depths_verts_i.append(pc_cam_list[mesh_i][:, 2:3])
                if "xyz" in mode:  # color,depth,xyz
                    colors_depths_verts_i.append(verts)
                colors_depths_verts_list.append(torch.cat(colors_depths_verts_i, dim=1))
            # [sum(Vi),C], C=1,3,4,or 7
            colors_depths_verts_packed, _ = _list_to_packed(colors_depths_verts_list)
            # render
            color_depth_xyz, _ = dr.interpolate(colors_depths_verts_packed, rast_out, faces_packed)
            if antialias:
                color_depth_xyz = dr.antialias(color_depth_xyz, rast_out, pos_clip_packed, faces_packed)

            if color_depth_xyz_code == [0, 0, 1]:  # 1
                ret["xyz"] = color_depth_xyz[..., :3]
            elif color_depth_xyz_code == [0, 1, 0]:  # 2
                ret["depth"] = color_depth_xyz[..., 0]
            elif color_depth_xyz_code == [0, 1, 1]:  # 3
                ret["depth"] = color_depth_xyz[..., 0]
                ret["xyz"] = color_depth_xyz[..., 1:4]
            elif color_depth_xyz_code == [1, 0, 0]:  # 4
                ret["color"] = color_depth_xyz[..., :3]
            elif color_depth_xyz_code == [1, 0, 1]:  # 5
                ret["color"] = color_depth_xyz[..., :3]
                ret["xyz"] = color_depth_xyz[..., 3:6]
            elif color_depth_xyz_code == [1, 1, 0]:  # 6
                ret["color"] = color_depth_xyz[..., :3]
                ret["depth"] = color_depth_xyz[..., 3]
            elif color_depth_xyz_code == [1, 1, 1]:  # 7
                ret["color"] = color_depth_xyz[..., :3]
                ret["depth"] = color_depth_xyz[..., 3]
                ret["xyz"] = color_depth_xyz[..., 4:7]

        # color: bhw3; mask: bhw; depth: bhw; xyz: bhw3
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
        rot_type="mat",
        uv_type="vertex",
        enable_mip=True,
        max_mip_level=10,
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
        assert len(mode) >= 1, mode
        bs = Rs.shape[0]
        if not isinstance(Ks, (tuple, list)) and Ks.ndim == 2:
            if isinstance(Ks, torch.Tensor):
                Ks = [Ks.clone() for _ in range(bs)]
            elif isinstance(Ks, np.ndarray):
                Ks = [Ks.copy() for _ in range(bs)]
            else:
                raise TypeError(f"Unknown type of Ks: {type(Ks)}")
        Ps = [self._projection_real(cam=K, x0=0, y0=0, w=width, h=height, nc=znear, fc=zfar) for K in Ks]
        Ps = torch.stack(Ps, dim=0)  # [b,4,4]
        # Modelview + projection matrix.
        mvp = torch.matmul(Ps, self._V)  # [b,4,4]
        assert rot_type in ["mat", "quat"], f"Unknown rot_type: {rot_type}"
        poses_4x4 = self._get_poses(Rs, ts, rot_type=rot_type)  # [b,4,4]
        mtx = torch.matmul(mvp, poses_4x4)  # [b,4,4]

        verts_list = [torch.squeeze(model["vertices"]) for model in models]
        faces_list = [torch.squeeze(model["faces"].to(torch.int32)) for model in models]
        meshes = Meshes(verts=verts_list, faces=faces_list)
        # verts_packed = meshes.verts_packed()  # [sum(Vi),3]
        faces_packed = meshes.faces_packed().to(dtype=torch.int32)  # [sum(Fi),3]
        faces_ranges = torch.stack(
            [
                meshes.mesh_to_faces_packed_first_idx(),
                meshes.num_faces_per_mesh(),
            ],
            dim=1,
        ).to(dtype=torch.int32, device="cpu")

        ####### render ###############
        # list of [Vi, 4]
        pos_clip_list = [self._transform_pos(mtx_i, _v)[0] for mtx_i, _v in zip(mtx, verts_list)]
        pos_clip_packed, _ = _list_to_packed(pos_clip_list)  # [sum(Vi),4]

        rast_out, rast_out_db = dr.rasterize(
            self._glctx,
            pos_clip_packed,
            faces_packed,
            ranges=faces_ranges,
            resolution=[height, width],
        )
        mask = torch.clamp(rast_out[..., -1:], 0, 1)
        ret = {}
        if "mask" in mode:
            ret["mask"] = mask.squeeze(-1)
        if "color" in mode:
            assert uv_type in ["vertex", "face"], uv_type
            if uv_type == "vertex":
                uv_list = [torch.squeeze(model["vertex_uvs"]) for model in models]
                uv_packed, _ = _list_to_packed(uv_list)
                uv_idx_packed = faces_packed  # faces
            else:  # face uv
                uv_list = [torch.squeeze(model["face_uvs"]) for model in models]
                uv_packed, _ = _list_to_packed(uv_list)
                uv_idx_list = [torch.squeeze(model["face_uv_ids"]).to(dtype=torch.int32) for model in models]
                uv_idx_packed, uv_idx_ranges = _list_to_packed(uv_idx_list)
            # NOTE: must be the same size
            tex_list = [torch.squeeze(model["texture"]) for model in models]
            tex_batch = torch.stack(tex_list, dim=0)  # [b,H,W,3]

            if enable_mip:
                texc, texd = dr.interpolate(
                    uv_packed,
                    rast_out,
                    uv_idx_packed,
                    rast_db=rast_out_db,
                    diff_attrs=self._diff_attrs,
                )
                color = dr.texture(
                    tex_batch,
                    texc,
                    texd,
                    filter_mode="linear-mipmap-linear",
                    max_mip_level=max_mip_level,
                )
            else:
                texc, _ = dr.interpolate(uv_packed, rast_out, uv_idx_packed)
                color = dr.texture(tex_batch, texc, filter_mode="linear")

            color = color * mask  # Mask out background.
            ret["color"] = color

        depth_xyz_code = _get_depth_xyz_code(mode)
        if sum(depth_xyz_code) > 0:
            if "depth" in mode:
                pc_cam_list = [misc.transform_pts_Rt_th(_v, R, t) for _v, R, t in zip(verts_list, Rs, ts)]
            else:
                pc_cam_list = None
            depths_verts_list = []
            for mesh_i, verts in enumerate(verts_list):
                depths_verts_i = []
                if "depth" in mode:
                    depths_verts_i.append(pc_cam_list[mesh_i][:, 2:3])
                if "xyz" in mode:
                    depths_verts_i.append(verts)
                depths_verts_list.append(torch.cat(depths_verts_i, dim=1))
            # [sum(Vi),C], C=1,3,or 4
            depths_verts_packed, _ = _list_to_packed(depths_verts_list)

            depth_xyz, _ = dr.interpolate(depths_verts_packed, rast_out, faces_packed)
            depth_xyz = depth_xyz * mask  # Mask out background.
            if depth_xyz_code == [0, 1]:  # 1
                ret["xyz"] = depth_xyz[..., :3]
            elif depth_xyz_code == [1, 0]:  # 2
                ret["depth"] = depth_xyz[..., 0]
            elif depth_xyz_code == [1, 1]:  # 3
                ret["depth"] = depth_xyz[..., 0]
                ret["xyz"] = depth_xyz[..., 1:4]
        # color: bhw3; mask: bhw; depth: bhw; xyz: bhw3
        return ret

    def render_batch_single(
        self,
        Rs,
        ts,
        model,
        *,
        Ks,
        width,
        height,
        znear=0.01,
        zfar=100,
        rot_type="mat",
        antialias=True,
        mode=["color", "depth"],
    ):
        """render a batch (vertex color) for the same object
        Args:
            Rs (tensor): [b,3,3] or [b,4]
            ts (tensor): [b,3,]
            model (dict): stores {"vertices":, "colors":, "faces":, }
            Ks (ndarray): [b,3,3] or [3,3]
            mode: color, depth, mask, xyz (one or more must be given)
        Returns:
            dict:
                color: bhw3
                mask: bhw
                depth: bhw
                xyz: bhw3
        """
        assert len(mode) >= 1, mode
        bs = Rs.shape[0]
        if not isinstance(Ks, (tuple, list)) and Ks.ndim == 2:
            if isinstance(Ks, torch.Tensor):
                Ks = [Ks.clone() for _ in range(bs)]
            elif isinstance(Ks, np.ndarray):
                Ks = [Ks.copy() for _ in range(bs)]
            else:
                raise TypeError(f"Unknown type of Ks: {type(Ks)}")
        Ps = [self._projection_real(cam=K, x0=0, y0=0, w=width, h=height, nc=znear, fc=zfar) for K in Ks]
        Ps = torch.stack(Ps, dim=0)  # [b,4,4]
        # Modelview + projection matrix.
        mvp = torch.matmul(Ps, self._V)  # [b,4,4]
        assert rot_type in ["mat", "quat"], f"Unknown rot_type: {rot_type}"
        poses_4x4 = self._get_poses(Rs, ts, rot_type=rot_type)  # [b,4,4]
        mtx = torch.matmul(mvp, poses_4x4)  # [b,4,4]

        verts = torch.squeeze(model["vertices"])
        faces = torch.squeeze(model["faces"].to(torch.int32))

        # color, depth, xyz
        if "color" in mode:
            model_colors = torch.squeeze(model["colors"])
        else:
            model_colors = None
        if "depth" in mode:
            pc_cam_list = [misc.transform_pts_Rt_th(verts, R, t) for R, t in zip(Rs, ts)]
        else:
            pc_cam_list = None

        ####### render the batch --------------------
        # list of [V, 4]
        pos_clip_list = [self._transform_pos(mtx_i, verts)[0] for mtx_i in mtx]
        pos_clip_batch = torch.stack(pos_clip_list, dim=0)  # [b,V,4]

        rast_out, _ = dr.rasterize(self._glctx, pos_clip_batch, faces, resolution=[height, width])
        ret = {}
        if "mask" in mode:
            mask = torch.clamp(rast_out[:, :, :, -1], 0, 1)
            ret["mask"] = mask
        color_depth_xyz_code = _get_color_depth_xyz_code(mode)
        if sum(color_depth_xyz_code) > 0:
            if color_depth_xyz_code == [0, 0, 1]:
                colors_depths_verts_batch = verts[None]  # [1,V,3]
            elif color_depth_xyz_code == [1, 0, 0]:
                colors_depths_verts_batch = model_colors[None]  # [1,V,3]
            elif color_depth_xyz_code == [1, 0, 1]:
                colors_depths_verts_batch = torch.cat([model_colors, verts], dim=1)[None]  # [1,V,6]
            else:
                # list of [V, C], C=1,4,or 7
                colors_depths_verts_list = []
                for b_i in range(bs):
                    colors_depths_verts_i = []
                    if "color" in mode:
                        colors_depths_verts_i.append(model_colors)
                    if "depth" in mode:
                        colors_depths_verts_i.append(pc_cam_list[b_i][:, 2:3])
                    if "xyz" in mode:
                        colors_depths_verts_i.append(verts)
                    colors_depths_verts_list.append(torch.cat(colors_depths_verts_i, dim=1))
                colors_depths_verts_batch = torch.stack(colors_depths_verts_list, dim=0)  # [b,V,C]

            color_depth_xyz, _ = dr.interpolate(colors_depths_verts_batch, rast_out, faces)
            if antialias:
                color_depth_xyz = dr.antialias(color_depth_xyz, rast_out, pos_clip_batch, faces)

            if color_depth_xyz_code == [0, 0, 1]:  # 1
                ret["xyz"] = color_depth_xyz[..., :3]
            elif color_depth_xyz_code == [0, 1, 0]:  # 2
                ret["depth"] = color_depth_xyz[..., 0]
            elif color_depth_xyz_code == [0, 1, 1]:  # 3
                ret["depth"] = color_depth_xyz[..., 0]
                ret["xyz"] = color_depth_xyz[..., 1:4]
            elif color_depth_xyz_code == [1, 0, 0]:  # 4
                ret["color"] = color_depth_xyz[..., :3]
            elif color_depth_xyz_code == [1, 0, 1]:  # 5
                ret["color"] = color_depth_xyz[..., :3]
                ret["xyz"] = color_depth_xyz[..., 3:6]
            elif color_depth_xyz_code == [1, 1, 0]:  # 6
                ret["color"] = color_depth_xyz[..., :3]
                ret["depth"] = color_depth_xyz[..., 3]
            elif color_depth_xyz_code == [1, 1, 1]:  # 7
                ret["color"] = color_depth_xyz[..., :3]
                ret["depth"] = color_depth_xyz[..., 3]
                ret["xyz"] = color_depth_xyz[..., 4:7]
        # color: bhw3; mask: bhw; depth: bhw; xyz: bhw3
        return ret

    def render_batch_single_tex(
        self,
        Rs,
        ts,
        model,
        *,
        Ks,
        width,
        height,
        znear=0.01,
        zfar=100,
        rot_type="mat",
        uv_type="vertex",
        enable_mip=True,
        max_mip_level=10,
        mode=["color", "depth"],
    ):
        """render a batch for a same textured object
        Args:
            Rs: [b,3,3] or [b,4] tensor
            ts: [b,3] tensor
            model: stores
                vertex uv: {"vertices":, "faces":, "texture":, "vertex_uvs":,}
                or face uv: {"vertices":, "faces":, "texture":, "face_uvs":, "face_uv_ids":,}
            Ks: [b,3,3] or [3,3]
            uv_type: `vertex` | `face`
            mode: color, depth, mask, xyz (one or more must be given)
        Returns:
            dict:
                color: bhw3
                mask: bhw
                depth: bhw
        """
        assert len(mode) >= 1, mode
        bs = Rs.shape[0]
        if not isinstance(Ks, (tuple, list)) and Ks.ndim == 2:
            if isinstance(Ks, torch.Tensor):
                Ks = [Ks.clone() for _ in range(bs)]
            elif isinstance(Ks, np.ndarray):
                Ks = [Ks.copy() for _ in range(bs)]
            else:
                raise TypeError(f"Unknown type of Ks: {type(Ks)}")
        Ps = [self._projection_real(cam=K, x0=0, y0=0, w=width, h=height, nc=znear, fc=zfar) for K in Ks]
        Ps = torch.stack(Ps, dim=0)  # [b,4,4]
        # Modelview + projection matrix.
        mvp = torch.matmul(Ps, self._V)  # [b,4,4]
        assert rot_type in ["mat", "quat"], f"Unknown rot_type: {rot_type}"
        poses_4x4 = self._get_poses(Rs, ts, rot_type=rot_type)  # [b,4,4]
        mtx = torch.matmul(mvp, poses_4x4)  # [b,4,4]

        verts = torch.squeeze(model["vertices"])
        faces = torch.squeeze(model["faces"].to(torch.int32))

        ####### render ###############
        pos_clip_list = [self._transform_pos(mtx_i, verts)[0] for mtx_i in mtx]  # list of [V, 4]
        pos_clip_batch = torch.stack(pos_clip_list, dim=0)  # [b,V,4]

        assert uv_type in ["vertex", "face"], uv_type
        if uv_type == "vertex":
            uv = torch.squeeze(model["vertex_uvs"])
            uv_idx = faces  # faces
        else:  # face uv
            uv = torch.squeeze(model["face_uvs"])
            uv_idx = torch.squeeze(model["face_uv_ids"]).to(dtype=torch.int32)

        tex = torch.squeeze(model["texture"])  # [H,W,3]

        # Render as a batch
        rast_out, rast_out_db = dr.rasterize(self._glctx, pos_clip_batch, faces, resolution=[height, width])
        mask = torch.clamp(rast_out[..., -1:], 0, 1)  # bhw1
        ret = {}
        if "mask" in mode:
            ret["mask"] = mask.squeeze(-1)
        if "color" in mode:
            if enable_mip and self.output_db:
                texc, texd = dr.interpolate(
                    uv[None],
                    rast_out,
                    uv_idx,
                    rast_db=rast_out_db,
                    diff_attrs=self._diff_attrs,
                )
                color = dr.texture(
                    tex[None],
                    texc,
                    uv_da=texd,
                    filter_mode="linear-mipmap-linear",
                    max_mip_level=max_mip_level,
                )
            else:
                texc, _ = dr.interpolate(uv[None], rast_out, uv_idx)
                color = dr.texture(tex[None], texc, filter_mode="linear")

            color = color * mask  # Mask out background.
            ret["color"] = color

        depth_xyz_code = _get_depth_xyz_code(mode)
        if sum(depth_xyz_code) > 0:
            if "depth" in mode:
                pc_cam_list = [misc.transform_pts_Rt_th(verts, R, t) for R, t in zip(Rs, ts)]
            else:
                pc_cam_list = None
            if depth_xyz_code == [0, 1]:
                depths_verts_batch = verts[None]  # [1,V,3]
            else:
                depths_verts_list = []
                for b_i in range(bs):
                    depths_verts_i = []
                    if "depth" in mode:
                        depths_verts_i.append(pc_cam_list[b_i][:, 2:3])
                    if "xyz" in mode:
                        depths_verts_i.append(verts)
                    depths_verts_list.append(torch.cat(depths_verts_i, dim=1))
                # [b,V,C], C=1 or 4
                depths_verts_batch = torch.stack(depths_verts_list, dim=0)

            depth_xyz, _ = dr.interpolate(depths_verts_batch, rast_out, faces)
            depth_xyz = depth_xyz * mask  # Mask out background.

            if depth_xyz_code == [0, 1]:  # 1
                ret["xyz"] = depth_xyz[..., :3]
            elif depth_xyz_code == [1, 0]:  # 2
                ret["depth"] = depth_xyz[..., 0]
            elif depth_xyz_code == [1, 1]:  # 3
                ret["depth"] = depth_xyz[..., 0]
                ret["xyz"] = depth_xyz[..., 1:4]
        # color: bhw3; mask: bhw; depth: bhw; xyz: bhw3
        return ret
