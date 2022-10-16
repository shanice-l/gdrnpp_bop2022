import os.path as osp
import torch
import numpy as np
import mmcv
import itertools
from einops import rearrange
from lib.egl_renderer.egl_renderer_v3 import EGLRenderer
from core.utils.camera_geometry import get_K_crop_resize
from core.utils.data_utils import xyz_to_region_batch
from lib.vis_utils.image import grid_show
from core.utils.utils import get_emb_show
from lib.pysixd import misc


def batch_data(cfg, data, renderer=None, device="cuda", phase="train"):
    if phase != "train":
        return batch_data_test(cfg, data, device=device)

    if cfg.MODEL.POSE_NET.XYZ_ONLINE:
        assert renderer is not None, "renderer must be provided for online rendering"
        return batch_data_train_online(cfg, data, renderer=renderer, device=device)

    # batch training data
    batch = {}
    batch["roi_img"] = torch.stack([d["roi_img"] for d in data], dim=0).to(device, non_blocking=True)
    if cfg.INPUT.WITH_DEPTH:
        batch["roi_depth"] = torch.stack([d["roi_depth"] for d in data], dim=0).to(device, non_blocking=True)

    batch["roi_cls"] = torch.as_tensor([d["roi_cls"] for d in data], dtype=torch.long).to(device, non_blocking=True)
    if "roi_coord_2d" in data[0]:
        batch["roi_coord_2d"] = torch.stack([d["roi_coord_2d"] for d in data], dim=0).to(
            device=device, non_blocking=True
        )

    if "roi_coord_2d_rel" in data[0]:
        batch["roi_coord_2d_rel"] = torch.stack([d["roi_coord_2d_rel"] for d in data], dim=0).to(
            device=device, non_blocking=True
        )

    batch["roi_cam"] = torch.stack([d["cam"] for d in data], dim=0).to(device, non_blocking=True)
    batch["roi_center"] = torch.stack([d["bbox_center"] for d in data], dim=0).to(
        device=device, dtype=torch.float32, non_blocking=True
    )
    batch["roi_wh"] = torch.stack([d["roi_wh"] for d in data], dim=0).to(device, non_blocking=True)
    batch["resize_ratio"] = torch.as_tensor([d["resize_ratio"] for d in data]).to(
        device=device, dtype=torch.float32, non_blocking=True
    )
    batch["roi_extent"] = torch.stack([d["roi_extent"] for d in data], dim=0).to(
        device=device, dtype=torch.float32, non_blocking=True
    )

    batch["roi_trans_ratio"] = torch.stack([d["trans_ratio"] for d in data], dim=0).to(device, non_blocking=True)
    # yapf: disable
    for key in [
        "roi_xyz", "roi_xyz_bin",
        "roi_mask_trunc", "roi_mask_visib", "roi_mask_obj", "roi_mask_full",
        "roi_region",
        "ego_rot", "trans",
        "roi_points",
    ]:
        if key in data[0]:
            if key in ["roi_region"]:
                dtype = torch.long
            else:
                dtype = torch.float32
            batch[key] = torch.stack([d[key] for d in data], dim=0).to(
                device=device, dtype=dtype, non_blocking=True
            )
    # yapf: enable
    if "sym_info" in data[0]:
        batch["sym_info"] = [d["sym_info"] for d in data]

    return batch


def batch_data_train_online(cfg, data, renderer, device="cuda"):
    # batch training data, rendering xyz online
    net_cfg = cfg.MODEL.POSE_NET
    g_head_cfg = net_cfg.GEO_HEAD
    batch = {}
    batch["roi_img"] = torch.stack([d["roi_img"] for d in data], dim=0).to(device, non_blocking=True)
    if cfg.INPUT.WITH_DEPTH:
        batch["roi_depth"] = torch.stack([d["roi_depth"] for d in data], dim=0).to(device, non_blocking=True)
    batch["roi_cls"] = torch.as_tensor([d["roi_cls"] for d in data], dtype=torch.long).to(device, non_blocking=True)
    bs = batch["roi_cls"].shape[0]
    if "roi_coord_2d" in data[0]:
        batch["roi_coord_2d"] = torch.stack([d["roi_coord_2d"] for d in data], dim=0).to(
            device=device, non_blocking=True
        )

    if "roi_coord_2d_rel" in data[0]:
        batch["roi_coord_2d_rel"] = torch.stack([d["roi_coord_2d_rel"] for d in data], dim=0).to(
            device=device, non_blocking=True
        )

    batch["roi_cam"] = torch.stack([d["cam"] for d in data], dim=0).to(device, non_blocking=True)
    batch["roi_center"] = torch.stack([d["bbox_center"] for d in data], dim=0).to(
        device=device, dtype=torch.float32, non_blocking=True
    )
    batch["roi_scale"] = torch.as_tensor([d["scale"] for d in data], device=device, dtype=torch.float32)
    batch["resize_ratio"] = torch.as_tensor(
        [d["resize_ratio"] for d in data], device=device, dtype=torch.float32
    )  # out_res/scale
    # get crop&resized K -------------------------------------------
    roi_crop_xy_batch = batch["roi_center"] - batch["roi_scale"].view(bs, -1) / 2
    out_res = net_cfg.OUTPUT_RES
    roi_resize_ratio_batch = out_res / batch["roi_scale"].view(bs, -1)
    batch["roi_zoom_K"] = get_K_crop_resize(batch["roi_cam"], roi_crop_xy_batch, roi_resize_ratio_batch)
    # --------------------------------------------------------------
    batch["roi_wh"] = torch.stack([d["roi_wh"] for d in data], dim=0).to(device, non_blocking=True)
    batch["roi_extent"] = torch.stack([d["roi_extent"] for d in data], dim=0).to(
        device=device, dtype=torch.float32, non_blocking=True
    )  # [b,3]

    batch["roi_trans_ratio"] = torch.stack([d["trans_ratio"] for d in data], dim=0).to(device, non_blocking=True)
    # yapf: disable
    for key in [
        "roi_mask_trunc", "roi_mask_visib", "roi_mask_full",
        "ego_rot", "trans",
        "roi_points",
    ]:
        if key in data[0]:
            dtype = torch.float32
            batch[key] = torch.stack([d[key] for d in data], dim=0).to(
                device=device, dtype=dtype, non_blocking=True
            )
    # yapf: enable
    if "sym_info" in data[0]:
        batch["sym_info"] = [d["sym_info"] for d in data]

    # rendering online xyz -----------------------------
    if net_cfg.XYZ_BP:
        pc_cam_tensor = torch.cuda.FloatTensor(out_res, out_res, 4, device=device).detach()
        roi_depth_batch = torch.empty(bs, out_res, out_res, dtype=torch.float32, device=device)
        for _i in range(bs):
            pose = np.hstack(
                [
                    batch["ego_rot"][_i].detach().cpu().numpy(),
                    batch["trans"][_i].detach().cpu().numpy().reshape(3, 1),
                ]
            )
            renderer.render(
                [int(batch["roi_cls"][_i])],
                [pose],
                K=batch["roi_zoom_K"][_i].detach().cpu().numpy(),
                pc_cam_tensor=pc_cam_tensor,
            )
            roi_depth_batch[_i].copy_(pc_cam_tensor[:, :, 2], non_blocking=True)
        roi_xyz_batch = misc.calc_xyz_bp_batch(
            roi_depth_batch,
            batch["ego_rot"],
            batch["trans"],
            batch["roi_zoom_K"],
            fmt="BHWC",
        )
    else:  # directly rendering xyz
        pc_obj_tensor = torch.cuda.FloatTensor(out_res, out_res, 4, device=device).detach()  # xyz
        roi_xyz_batch = torch.empty(bs, out_res, out_res, 3, dtype=torch.float32, device=device)
        for _i in range(bs):
            pose = np.hstack(
                [
                    batch["ego_rot"][_i].detach().cpu().numpy(),
                    batch["trans"][_i].detach().cpu().numpy().reshape(3, 1),
                ]
            )
            renderer.render(
                [int(batch["roi_cls"][_i])],
                [pose],
                K=batch["roi_zoom_K"][_i].detach().cpu().numpy(),
                pc_obj_tensor=pc_obj_tensor,
            )
            roi_xyz_batch[_i].copy_(pc_obj_tensor[:, :, :3], non_blocking=True)

    # [bs, out_res, out_res]
    batch["roi_mask_obj"] = (
        (roi_xyz_batch[..., 0] != 0) & (roi_xyz_batch[..., 1] != 0) & (roi_xyz_batch[..., 2] != 0)
    ).to(torch.float32)
    batch["roi_mask_trunc"] = batch["roi_mask_trunc"] * batch["roi_mask_obj"]
    batch["roi_mask_visib"] = batch["roi_mask_visib"] * batch["roi_mask_obj"]

    if g_head_cfg.NUM_REGIONS > 1:  # get roi_region ------------------------
        batch["roi_fps_points"] = torch.stack([d["roi_fps_points"] for d in data], dim=0).to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        batch["roi_region"] = xyz_to_region_batch(roi_xyz_batch, batch["roi_fps_points"], mask=batch["roi_mask_obj"])
    # normalize to [0, 1]
    batch["roi_xyz"] = rearrange(roi_xyz_batch, "b h w c -> b c h w") / batch["roi_extent"].view(bs, 3, 1, 1) + 0.5

    # get xyz bin if needed ---------------------------------
    loss_cfg = net_cfg.LOSS_CFG
    xyz_loss_type = loss_cfg.XYZ_LOSS_TYPE
    if ("CE" in xyz_loss_type) or ("cls" in net_cfg.NAME):
        # coordinates: [0, 1] to discrete [0, XYZ_BIN-1]
        roi_xyz_bin_batch = (
            (batch["roi_xyz"] * (g_head_cfg.XYZ_BIN - 1) + 0.5).clamp(min=0, max=g_head_cfg.XYZ_BIN).to(torch.long)
        )
        # set bg to XYZ_BIN
        roi_masks = {
            "trunc": batch["roi_mask_trunc"],
            "visib": batch["roi_mask_visib"],
            "obj": batch["roi_mask_obj"],
        }
        roi_mask_xyz = roi_masks[loss_cfg.XYZ_LOSS_MASK_GT]
        for _c in range(roi_xyz_bin_batch.shape[1]):
            roi_xyz_bin_batch[:, _c][roi_mask_xyz == 0] = g_head_cfg.XYZ_BIN
        batch["roi_xyz_bin"] = roi_xyz_bin_batch

    if cfg.TRAIN.VIS:
        vis_batch(cfg, batch, phase="train")
    return batch


def batch_data_test(cfg, data, device="cuda"):
    batch = {}

    # yapf: disable
    roi_keys = ["im_H", "im_W",
                "roi_img", "inst_id", "roi_coord_2d", "roi_coord_2d_rel",
                "roi_cls", "score", "time", "roi_extent",
                "bbox", "bbox_est", "bbox_mode", "roi_wh",
                "scale", "resize_ratio",
    ]
    if cfg.INPUT.WITH_DEPTH:
        roi_keys.append("roi_depth")
    for key in roi_keys:
        if key in ["roi_cls"]:
            dtype = torch.long
        else:
            dtype = torch.float32
        if key in data[0]:
            batch[key] = torch.cat([d[key] for d in data], dim=0).to(device=device, dtype=dtype, non_blocking=True)
    # yapf: enable

    batch["roi_cam"] = torch.cat([d["cam"] for d in data], dim=0).to(device, non_blocking=True)
    batch["roi_center"] = torch.cat([d["bbox_center"] for d in data], dim=0).to(device, non_blocking=True)
    for key in ["scene_im_id", "file_name", "model_info"]:
        # flatten the lists
        if key in data[0]:
            batch[key] = list(itertools.chain(*[d[key] for d in data]))

    return batch

def batch_data_inference_roi(cfg, data, device='cuda'):
    net_cfg = cfg.MODEL.POSE_NET
    g_head_cfg = net_cfg.GEO_HEAD
    batch = {}
    batch["roi_img"] = torch.cat([d["roi_img"] for d in data], dim=0).to(device, non_blocking=True)
    bs = batch["roi_img"].shape[0]


    batch["roi_cam"] = torch.cat([d["cam"] for d in data], dim=0).to(device, non_blocking=True)
    batch["roi_center"] = torch.cat([d["bbox_center"] for d in data], dim=0).to(
        device=device, dtype=torch.float32, non_blocking=True
    )
    batch["roi_scale"] = [torch.as_tensor(d["scale"], device=device, dtype=torch.float32) for d in data]
    batch["roi_scale"] = torch.cat(batch["roi_scale"], dim=0).to(
        device=device, dtype=torch.float32, non_blocking=True)
    batch["resize_ratio"] = [torch.as_tensor(d["resize_ratio"], device=device, dtype=torch.float32) for d in data]  # out_res/scale
    batch["resize_ratio"] = torch.cat(batch["resize_ratio"], dim=0).to(
        device=device, dtype=torch.float32, non_blocking=True)
    # get crop&resized K -------------------------------------------
    roi_crop_xy_batch = batch["roi_center"] - batch["roi_scale"].view(bs, -1) / 2
    out_res = net_cfg.OUTPUT_RES
    roi_resize_ratio_batch = out_res / batch["roi_scale"].view(bs, -1)
    batch["roi_zoom_K"] = get_K_crop_resize(batch["roi_cam"], roi_crop_xy_batch, roi_resize_ratio_batch)
    return batch


def get_renderer(cfg, data_ref, obj_names, gpu_id=None):
    """for rendering the targets (xyz) online."""
    model_dir = data_ref.model_dir

    obj_ids = [data_ref.obj2id[_obj] for _obj in obj_names]
    model_paths = [osp.join(model_dir, "obj_{:06d}.ply".format(obj_id)) for obj_id in obj_ids]

    texture_paths = None
    if data_ref.texture_paths is not None:
        texture_paths = [osp.join(model_dir, "obj_{:06d}.png".format(obj_id)) for obj_id in obj_ids]

    ren = EGLRenderer(
        model_paths,
        texture_paths=texture_paths,
        vertex_scale=data_ref.vertex_scale,
        znear=data_ref.zNear,
        zfar=data_ref.zFar,
        K=data_ref.camera_matrix,  # may override later
        height=cfg.MODEL.POSE_NET.OUTPUT_RES,
        width=cfg.MODEL.POSE_NET.OUTPUT_RES,
        gpu_id=gpu_id,
        use_cache=True,
    )
    return ren


def get_out_coor(cfg, coor_x, coor_y, coor_z):
    if (coor_x.shape[1] == 1) and (coor_y.shape[1] == 1) and (coor_z.shape[1] == 1):
        coor_ = torch.cat([coor_x, coor_y, coor_z], dim=1)
    else:
        coor_ = torch.stack(
            [
                torch.argmax(coor_x, dim=1),
                torch.argmax(coor_y, dim=1),
                torch.argmax(coor_z, dim=1),
            ],
            dim=1,
        )
        # set the coordinats of background to (0, 0, 0)
        coor_[coor_ == cfg.MODEL.POSE_NET.GEO_HEAD.XYZ_BIN] = 0
        # normalize the coordinates to [0, 1]
        coor_ = coor_ / float(cfg.MODEL.POSE_NET.GEO_HEAD.XYZ_BIN - 1)

    return coor_


def get_out_mask(cfg, pred_mask):
    # (b,c,h,w)
    # output: (b, 1, h, w)
    mask_loss_type = cfg.MODEL.POSE_NET.LOSS_CFG.MASK_LOSS_TYPE
    bs, c, h, w = pred_mask.shape
    if mask_loss_type == "L1":
        assert c == 1, c
        mask_max = torch.max(pred_mask.view(bs, -1), dim=-1)[0].view(bs, 1, 1, 1)
        mask_min = torch.min(pred_mask.view(bs, -1), dim=-1)[0].view(bs, 1, 1, 1)
        # [0, 1]
        out_mask = (pred_mask - mask_min) / (mask_max - mask_min)  # + 1e-6)
    elif mask_loss_type in ["BCE", "RW_BCE", "dice"]:
        assert c == 1, c
        out_mask = torch.sigmoid(pred_mask)
    elif mask_loss_type == "CE":
        out_mask = torch.argmax(pred_mask, dim=1, keepdim=True)
    else:
        raise NotImplementedError(f"unknown mask loss type: {mask_loss_type}")
    return out_mask


def vis_batch(cfg, batch, phase="train"):
    n_obj = batch["roi_cls"].shape[0]
    # yapf: disable
    for i in range(n_obj):
        vis_dict = {"roi_img": (batch['roi_img'][i].detach().cpu().numpy().transpose(1, 2, 0) * 255).astype('uint8')[:,:,::-1]}
        if phase == 'train':
            vis_dict['roi_mask_trunc'] = batch['roi_mask_trunc'][i].detach().cpu().numpy()
            vis_dict['roi_mask_visib'] = batch['roi_mask_visib'][i].detach().cpu().numpy()
            vis_dict['roi_mask_obj'] = batch['roi_mask_obj'][i].detach().cpu().numpy()

            vis_dict['roi_xyz'] = get_emb_show(batch['roi_xyz'][i].detach().cpu().numpy().transpose(1, 2, 0))

            if "roi_depth" in batch:
                vis_dict['roi_depth'] = get_emb_show(batch['roi_depth'][i].detach().cpu().numpy().transpose(1, 2, 0)[:, :, -1])

            roi_xyz_img_size = mmcv.imresize_like((vis_dict['roi_xyz'] * 255).astype("uint8"), vis_dict["roi_img"])

            vis_dict["roi_img_xyz"] = (vis_dict["roi_img"] * 0.5  + roi_xyz_img_size * 0.5).astype("uint8")

        show_titles = list(vis_dict.keys())
        show_ims = list(vis_dict.values())
        ncol = 4
        nrow = int(np.ceil(len(show_ims) / ncol))
        grid_show(show_ims, show_titles, row=nrow, col=ncol)
    # yapf: enable
