import copy
import torch
import torch.nn as nn
import numpy as np
from lib.pysixd.pose_error import re, te
from core.utils.pose_utils import quat2mat_torch
from core.utils.rot_reps import rot6d_to_mat_batch
from core.utils import lie_algebra, quaternion_lf
from .net_factory import NECKS, HEADS, FUSENETS


def get_xyz_doublemask_region_out_dim(cfg):
    net_cfg = cfg.MODEL.POSE_NET
    g_head_cfg = net_cfg.GEO_HEAD
    loss_cfg = net_cfg.LOSS_CFG

    xyz_loss_type = loss_cfg.XYZ_LOSS_TYPE
    if xyz_loss_type in ["MSE", "L1", "L2", "SmoothL1"]:
        xyz_out_dim = 3
    elif xyz_loss_type in ["CE_coor", "CE"]:
        xyz_out_dim = 3 * (g_head_cfg.XYZ_BIN + 1)
    else:
        raise NotImplementedError(f"unknown xyz loss type: {xyz_loss_type}")

    mask_loss_type = loss_cfg.MASK_LOSS_TYPE
    if mask_loss_type in ["L1", "BCE", "RW_BCE", "dice"]:
        mask_out_dim = 2
    elif mask_loss_type in ["CE"]:
        mask_out_dim = 4
    else:
        raise NotImplementedError(f"unknown mask loss type: {mask_loss_type}")

    region_out_dim = g_head_cfg.NUM_REGIONS + 1
    # at least 2 regions (with bg, at least 3 regions)
    assert region_out_dim > 2, region_out_dim

    return xyz_out_dim, mask_out_dim, region_out_dim


def get_xyz_mask_region_out_dim(cfg):
    net_cfg = cfg.MODEL.POSE_NET
    g_head_cfg = net_cfg.GEO_HEAD
    loss_cfg = net_cfg.LOSS_CFG

    xyz_loss_type = loss_cfg.XYZ_LOSS_TYPE
    if xyz_loss_type in ["MSE", "L1", "L2", "SmoothL1"]:
        xyz_out_dim = 3
    elif xyz_loss_type in ["CE_coor", "CE"]:
        xyz_out_dim = 3 * (g_head_cfg.XYZ_BIN + 1)
    else:
        raise NotImplementedError(f"unknown xyz loss type: {xyz_loss_type}")

    mask_loss_type = loss_cfg.MASK_LOSS_TYPE
    if mask_loss_type in ["L1", "BCE", "RW_BCE", "dice"]:
        mask_out_dim = 1
    elif mask_loss_type in ["CE"]:
        mask_out_dim = 2
    else:
        raise NotImplementedError(f"unknown mask loss type: {mask_loss_type}")

    region_out_dim = g_head_cfg.NUM_REGIONS + 1
    # at least 2 regions (with bg, at least 3 regions)
    assert region_out_dim > 2, region_out_dim

    return xyz_out_dim, mask_out_dim, region_out_dim


def get_xyz_mask_out_dim(cfg):
    net_cfg = cfg.MODEL.POSE_NET
    g_head_cfg = net_cfg.GEO_HEAD
    loss_cfg = net_cfg.LOSS_CFG

    xyz_loss_type = loss_cfg.XYZ_LOSS_TYPE
    mask_loss_type = loss_cfg.MASK_LOSS_TYPE
    if xyz_loss_type in ["MSE", "L1", "L2", "SmoothL1"]:
        r_out_dim = 3
    elif xyz_loss_type in ["CE_coor", "CE"]:
        r_out_dim = 3 * (g_head_cfg.XYZ_BIN + 1)
    else:
        raise NotImplementedError(f"unknown xyz loss type: {xyz_loss_type}")

    if mask_loss_type in ["L1", "BCE", "RW_BCE", "dice"]:
        mask_out_dim = 1
    elif mask_loss_type in ["CE"]:
        mask_out_dim = 2
    else:
        raise NotImplementedError(f"unknown mask loss type: {mask_loss_type}")

    return r_out_dim, mask_out_dim


def get_neck(cfg):
    net_cfg = cfg.MODEL.POSE_NET
    neck_cfg = net_cfg.NECK
    params_lr_list = []
    if neck_cfg.ENABLED:
        neck_init_cfg = copy.deepcopy(neck_cfg.INIT_CFG)
        neck_type = neck_init_cfg.pop("type")
        neck = NECKS[neck_type](**neck_init_cfg)
        if neck_cfg.FREEZE:
            for param in neck.parameters():
                with torch.no_grad():
                    param.requires_grad = False
        else:
            params_lr_list.append(
                {
                    "params": filter(lambda p: p.requires_grad, neck.parameters()),
                    "lr": float(cfg.SOLVER.BASE_LR) * neck_cfg.LR_MULT,
                }
            )
    else:
        neck = None
    return neck, params_lr_list


def get_geo_head(cfg):
    net_cfg = cfg.MODEL.POSE_NET
    geo_head_cfg = net_cfg.GEO_HEAD
    params_lr_list = []

    geo_head_init_cfg = copy.deepcopy(geo_head_cfg.INIT_CFG)
    geo_head_type = geo_head_init_cfg.pop("type")

    xyz_num_classes = net_cfg.NUM_CLASSES if geo_head_cfg.XYZ_CLASS_AWARE else 1
    mask_num_classes = net_cfg.NUM_CLASSES if geo_head_cfg.MASK_CLASS_AWARE else 1
    if geo_head_cfg.NUM_REGIONS <= 1:
        xyz_dim, mask_dim = get_xyz_mask_out_dim(cfg)
        geo_head_init_cfg.update(
            xyz_num_classes=xyz_num_classes,
            mask_num_classes=mask_num_classes,
            xyz_out_dim=xyz_dim,
            mask_out_dim=mask_dim,
        )
    elif "DoubleMask" in geo_head_type:
        xyz_dim, mask_dim, region_dim = get_xyz_doublemask_region_out_dim(cfg)
        region_num_classes = net_cfg.NUM_CLASSES if geo_head_cfg.REGION_CLASS_AWARE else 1
        geo_head_init_cfg.update(
            xyz_num_classes=xyz_num_classes,
            mask_num_classes=mask_num_classes,
            region_num_classes=region_num_classes,
            xyz_out_dim=xyz_dim,
            mask_out_dim=mask_dim,
            region_out_dim=region_dim,
        )
    else:
        xyz_dim, mask_dim, region_dim = get_xyz_mask_region_out_dim(cfg)
        region_num_classes = net_cfg.NUM_CLASSES if geo_head_cfg.REGION_CLASS_AWARE else 1
        geo_head_init_cfg.update(
            xyz_num_classes=xyz_num_classes,
            mask_num_classes=mask_num_classes,
            region_num_classes=region_num_classes,
            xyz_out_dim=xyz_dim,
            mask_out_dim=mask_dim,
            region_out_dim=region_dim,
        )
    geo_head = HEADS[geo_head_type](**geo_head_init_cfg)

    if geo_head_cfg.FREEZE:
        for param in geo_head.parameters():
            with torch.no_grad():
                param.requires_grad = False
    else:
        params_lr_list.append(
            {
                "params": filter(lambda p: p.requires_grad, geo_head.parameters()),
                "lr": float(cfg.SOLVER.BASE_LR) * geo_head_cfg.LR_MULT,
            }
        )

    return geo_head, params_lr_list


def get_fuse_net(cfg):
    net_cfg = cfg.MODEL.POSE_NET
    fuse_net_cfg = net_cfg.FUSE_NET

    fuse_net_init_cfg = copy.deepcopy(fuse_net_cfg.INIT_CFG)
    fuse_net_type = fuse_net_init_cfg.pop("type")

    fuse_net = FUSENETS[fuse_net_type](**fuse_net_init_cfg)

    params_lr_list = []
    if fuse_net_cfg.FREEZE:
        for param in fuse_net.parameters():
            with torch.no_grad():
                param.requires_grad = False
    else:
        params_lr_list.append(
            {
                "params": filter(lambda p: p.requires_grad, fuse_net.parameters()),
                "lr": float(cfg.SOLVER.BASE_LR) * fuse_net_cfg.LR_MULT,
            }
        )

    return fuse_net, params_lr_list


def get_pnp_net(cfg):
    net_cfg = cfg.MODEL.POSE_NET
    g_head_cfg = net_cfg.GEO_HEAD
    pnp_net_cfg = net_cfg.PNP_NET
    loss_cfg = net_cfg.LOSS_CFG

    xyz_dim, mask_dim, region_dim = get_xyz_mask_region_out_dim(cfg)

    if loss_cfg.XYZ_LOSS_TYPE in ["CE_coor", "CE"]:
        pnp_net_in_channel = xyz_dim - 3  # for bin xyz, no bg channel
    else:
        pnp_net_in_channel = xyz_dim

    if pnp_net_cfg.WITH_2D_COORD:
        pnp_net_in_channel += 2

    if pnp_net_cfg.REGION_ATTENTION:
        pnp_net_in_channel += g_head_cfg.NUM_REGIONS

    if pnp_net_cfg.MASK_ATTENTION in ["concat"]:  # do not add dim for none/mul
        pnp_net_in_channel += 1

    if pnp_net_cfg.ROT_TYPE in ["allo_quat", "ego_quat"]:
        rot_dim = 4
    elif pnp_net_cfg.ROT_TYPE in [
        "allo_log_quat",
        "ego_log_quat",
        "allo_lie_vec",
        "ego_lie_vec",
    ]:
        rot_dim = 3
    elif pnp_net_cfg.ROT_TYPE in ["allo_rot6d", "ego_rot6d"]:
        rot_dim = 6
    else:
        raise ValueError(f"Unknown ROT_TYPE: {pnp_net_cfg.ROT_TYPE}")

    pnp_net_init_cfg = copy.deepcopy(pnp_net_cfg.INIT_CFG)
    pnp_head_type = pnp_net_init_cfg.pop("type")

    if pnp_head_type in ["ConvPnPNet", "ConvPnPNetCls"]:
        pnp_net_init_cfg.update(
            nIn=pnp_net_in_channel,
            rot_dim=rot_dim,
            num_regions=g_head_cfg.NUM_REGIONS,
            mask_attention_type=pnp_net_cfg.MASK_ATTENTION,
        )
    elif pnp_head_type == "PointPnPNet":
        pnp_net_init_cfg.update(
            nIn=pnp_net_in_channel,
            rot_dim=rot_dim,
            num_regions=g_head_cfg.NUM_REGIONS,
        )
    elif pnp_head_type == "SimplePointPnPNet":
        pnp_net_init_cfg.update(
            nIn=pnp_net_in_channel,
            rot_dim=rot_dim,
            mask_attention_type=pnp_net_cfg.MASK_ATTENTION,
            # num_regions=g_head_cfg.NUM_REGIONS,
        )
    else:
        raise ValueError(f"Unknown pnp head type: {pnp_head_type}")

    pnp_net = HEADS[pnp_head_type](**pnp_net_init_cfg)

    params_lr_list = []
    if pnp_net_cfg.FREEZE:
        for param in pnp_net.parameters():
            with torch.no_grad():
                param.requires_grad = False
    else:
        params_lr_list.append(
            {
                "params": filter(lambda p: p.requires_grad, pnp_net.parameters()),
                "lr": float(cfg.SOLVER.BASE_LR) * pnp_net_cfg.LR_MULT,
            }
        )
    return pnp_net, params_lr_list


def get_pnp_net_no_region(cfg):
    net_cfg = cfg.MODEL.POSE_NET
    g_head_cfg = net_cfg.GEO_HEAD
    pnp_net_cfg = net_cfg.PNP_NET
    loss_cfg = net_cfg.LOSS_CFG

    xyz_dim, mask_dim = get_xyz_mask_out_dim(cfg)

    if loss_cfg.XYZ_LOSS_TYPE in ["CE_coor", "CE"]:
        pnp_net_in_channel = xyz_dim - 3  # for bin xyz, no bg channel
    else:
        pnp_net_in_channel = xyz_dim

    if pnp_net_cfg.WITH_2D_COORD:
        pnp_net_in_channel += 2

    if pnp_net_cfg.MASK_ATTENTION in ["concat"]:  # do not add dim for none/mul
        pnp_net_in_channel += 1

    if pnp_net_cfg.ROT_TYPE in ["allo_quat", "ego_quat"]:
        rot_dim = 4
    elif pnp_net_cfg.ROT_TYPE in [
        "allo_log_quat",
        "ego_log_quat",
        "allo_lie_vec",
        "ego_lie_vec",
    ]:
        rot_dim = 3
    elif pnp_net_cfg.ROT_TYPE in ["allo_rot6d", "ego_rot6d"]:
        rot_dim = 6
    else:
        raise ValueError(f"Unknown ROT_TYPE: {pnp_net_cfg.ROT_TYPE}")

    pnp_net_init_cfg = copy.deepcopy(pnp_net_cfg.INIT_CFG)
    pnp_head_type = pnp_net_init_cfg.pop("type")

    if pnp_head_type == "ConvPnPNetNoRegion":
        pnp_net_init_cfg.update(
            nIn=pnp_net_in_channel,
            rot_dim=rot_dim,
            mask_attention_type=pnp_net_cfg.MASK_ATTENTION,
        )
    elif pnp_head_type == "PointPnPNetNoRegion":
        pnp_net_init_cfg.update(nIn=pnp_net_in_channel, rot_dim=rot_dim)
    elif pnp_head_type == "SimplePointPnPNetNoRegion":
        pnp_net_init_cfg.update(
            nIn=pnp_net_in_channel,
            rot_dim=rot_dim,
            mask_attention_type=pnp_net_cfg.MASK_ATTENTION,
        )
    else:
        raise ValueError(f"Unknown pnp head type: {pnp_head_type}")

    pnp_net = HEADS[pnp_head_type](**pnp_net_init_cfg)

    params_lr_list = []
    if pnp_net_cfg.FREEZE:
        for param in pnp_net.parameters():
            with torch.no_grad():
                param.requires_grad = False
    else:
        params_lr_list.append(
            {
                "params": filter(lambda p: p.requires_grad, pnp_net.parameters()),
                "lr": float(cfg.SOLVER.BASE_LR) * pnp_net_cfg.LR_MULT,
            }
        )
    return pnp_net, params_lr_list


def get_rot_mat(rot, rot_type):
    if rot_type in ["ego_quat", "allo_quat"]:
        rot_m = quat2mat_torch(rot)
    elif rot_type in ["ego_log_quat", "allo_log_quat"]:
        # from latentfusion (lf)
        rot_m = quat2mat_torch(quaternion_lf.qexp(rot))
    elif rot_type in ["ego_lie_vec", "allo_lie_vec"]:
        rot_m = lie_algebra.lie_vec_to_rot(rot)
    elif rot_type in ["ego_rot6d", "allo_rot6d"]:
        rot_m = rot6d_to_mat_batch(rot)
    else:
        raise ValueError(f"Wrong pred_rot type: {rot_type}")
    return rot_m


def get_mask_prob(pred_mask, mask_loss_type):
    # (b,c,h,w)
    # output: (b, 1, h, w)
    bs, c, h, w = pred_mask.shape
    if mask_loss_type == "L1":
        assert c == 1, c
        mask_max = torch.max(pred_mask.view(bs, -1), dim=-1)[0].view(bs, 1, 1, 1)
        mask_min = torch.min(pred_mask.view(bs, -1), dim=-1)[0].view(bs, 1, 1, 1)
        # [0, 1]
        mask_prob = (pred_mask - mask_min) / (mask_max - mask_min)  # + 1e-6)
    elif mask_loss_type in ["BCE", "RW_BCE", "dice"]:
        assert c == 1, c
        mask_prob = torch.sigmoid(pred_mask)
    elif mask_loss_type == "CE":
        mask_prob = torch.softmax(pred_mask, dim=1, keepdim=True)[:, 1:2, :, :]
    else:
        raise NotImplementedError(f"Unknown mask loss type: {mask_loss_type}")
    return mask_prob


def compute_mean_re_te(pred_transes, pred_rots, gt_transes, gt_rots):
    pred_transes = pred_transes.detach().cpu().numpy()
    pred_rots = pred_rots.detach().cpu().numpy()
    gt_transes = gt_transes.detach().cpu().numpy()
    gt_rots = gt_rots.detach().cpu().numpy()

    bs = pred_rots.shape[0]
    R_errs = np.zeros((bs,), dtype=np.float32)
    T_errs = np.zeros((bs,), dtype=np.float32)
    for i in range(bs):
        R_errs[i] = re(pred_rots[i], gt_rots[i])
        T_errs[i] = te(pred_transes[i], gt_transes[i])
    return R_errs.mean(), T_errs.mean()
