import os.path as osp
import random
import sys

import cv2
import mmcv
import numpy as np
from tqdm import tqdm

cur_dir = osp.abspath(osp.dirname(__file__))
PROJ_ROOT = osp.join(cur_dir, "../../../..")
sys.path.insert(0, PROJ_ROOT)
from core.utils.data_utils import get_2d_coord_np
from lib.pysixd import inout, misc
from lib.pysixd.pose_error import calc_rt_dist_m
from lib.utils import logger
from lib.vis_utils.image import grid_show
from lib.utils.mask_utils import cocosegm2mask


random.seed(2333)

idx2class = {
    1: "ape",
    2: "benchvise",
    3: "bowl",
    4: "camera",
    5: "can",
    6: "cat",
    7: "cup",
    8: "driller",
    9: "duck",
    10: "eggbox",
    11: "glue",
    12: "holepuncher",
    13: "iron",
    14: "lamp",
    15: "phone",
}

class2idx = {_name: _id for _id, _name in idx2class.items()}

classes = idx2class.values()
classes = sorted(classes)

# DEPTH_FACTOR = 1000.
IM_H = 480
IM_W = 640
near = 0.01
far = 6.5

data_dir = osp.normpath(osp.join(PROJ_ROOT, "datasets/BOP_DATASETS/lm/train_egl"))

cls_indexes = [_idx for _idx in sorted(idx2class.keys())]
cls_names = [idx2class[cls_idx] for cls_idx in cls_indexes]
lm_model_dir = osp.normpath(osp.join(PROJ_ROOT, "datasets/BOP_DATASETS/lm/models"))
model_paths = [osp.join(lm_model_dir, f"obj_{cls_idx:06d}.ply") for cls_idx in cls_indexes]
texture_paths = None

xyz_root = osp.normpath(osp.join(data_dir, "xyz_crop"))
gt_path = osp.join(data_dir, "gt.json")
assert osp.exists(gt_path)

K = np.array([[572.4114, 0, 325.2611], [0, 573.57043, 242.04899], [0, 0, 1]])
DEPTH_FACTOR = 10000.0

coord2d = get_2d_coord_np(width=IM_W, height=IM_H, fmt="HWC")


def normalize_to_01(img):
    if img.max() != img.min():
        return (img - img.min()) / (img.max() - img.min())
    else:
        return img


def get_emb_show(bbox_emb):
    show_emb = bbox_emb.copy()
    show_emb = normalize_to_01(bbox_emb)
    return show_emb


def get_img_model_points_with_coords2d(mask_pred, xyz_pred, coord2d, im_H, im_W, max_num_points=-1, mask_thr=0.5):
    """
    from predicted crop_and_resized xyz, bbox top-left,
    get 2D-3D correspondences (image points, 3D model points)
    Args:
        mask_pred: HW, predicted mask in roi_size
        xyz_pred: HWC, predicted xyz in roi_size(eg. 64)
        coord2d: HW2 coords 2d in roi size
        im_H, im_W
        extent: size of x,y,z
    """
    coord2d = coord2d.copy()
    coord2d[:, :, 0] = coord2d[:, :, 0] * im_W
    coord2d[:, :, 1] = coord2d[:, :, 1] * im_H

    sel_mask = (
        (mask_pred > mask_thr)
        & (abs(xyz_pred[:, :, 0]) > 0.0001)
        & (abs(xyz_pred[:, :, 1]) > 0.0001)
        & (abs(xyz_pred[:, :, 2]) > 0.0001)
    )
    model_points = xyz_pred[sel_mask].reshape(-1, 3)
    image_points = coord2d[sel_mask].reshape(-1, 2)

    if max_num_points >= 4:
        num_points = len(image_points)
        max_keep = min(max_num_points, num_points)
        indices = [i for i in range(num_points)]
        random.shuffle(indices)
        model_points = model_points[indices[:max_keep]]
        image_points = image_points[indices[:max_keep]]
    return image_points, model_points


def get_pose_pnp_from_xyz_crop(emb_pred_, coord2d, im_H, im_W, K):
    # emb_pred_: emb_crop, HWC
    mask_pred = ((emb_pred_[:, :, 0] != 0) & (emb_pred_[:, :, 1] != 0) & (emb_pred_[:, :, 2] != 0)).astype("uint8")
    image_points, model_points = get_img_model_points_with_coords2d(mask_pred, emb_pred_, coord2d, im_H=im_H, im_W=im_W)
    pnp_method = cv2.SOLVEPNP_EPNP
    pose_est = misc.pnp_v2(
        model_points,
        image_points,
        K,
        method=pnp_method,
        ransac=True,
        ransac_reprojErr=3.0,
        ransac_iter=100,
    )
    return pose_est


class XyzVerify(object):
    def __init__(self):
        pass

    def main(self):
        gt_dict = mmcv.load(gt_path)
        r_errors = []
        t_errors = []
        for str_im_id, annos in tqdm(gt_dict.items()):
            int_im_id = int(str_im_id)
            im_path = osp.join(data_dir, f"rgb/{int_im_id:06d}.jpg")

            for anno_i, anno in enumerate(annos):
                obj_id = anno["obj_id"]
                pose = np.array(anno["pose"])

                mask = cocosegm2mask(anno["mask_full"], IM_H, IM_W)
                area = mask.sum()
                if area < 4:  # NOTE: pnp need at least 4 points
                    continue

                xyz_path = osp.join(xyz_root, f"{int_im_id:06d}_{anno_i:06d}-xyz.pkl")
                assert osp.exists(xyz_path), xyz_path
                xyz = np.zeros((height, width, 3), dtype=np.float32)
                xyz_info = mmcv.load(xyz_path)
                x1, y1, x2, y2 = xyz_info["xyxy"]
                w = x2 - x1 + 1
                h = y2 - y1 + 1
                bbox_area = w * h
                xyz[y1 : y2 + 1, x1 : x2 + 1, :] = xyz_info["xyz_crop"]
                num_xyz_point = (abs(xyz) > 1e-6).sum()
                if num_xyz_point < 4 or bbox_area < 4:
                    logger.warn(f"{xyz_path} num xyz point: {num_xyz_point} bbox_area: {bbox_area} mask_area: {area}")
                    continue

                coord2d_crop = coord2d[y1 : y2 + 1, x1 : x2 + 1, :]
                try:
                    pose_est = get_pose_pnp_from_xyz_crop(
                        xyz_info["xyz_crop"].astype("float32"),
                        coord2d_crop,
                        im_H=IM_H,
                        im_W=IM_W,
                        K=K,
                    )
                except:
                    pose_est = get_pose_pnp_from_xyz_crop(
                        xyz_info["xyz_crop"].astype("float32"),
                        coord2d_crop,
                        im_H=IM_H,
                        im_W=IM_W,
                        K=K,
                    )
                    logger.warn(f"{xyz_path} num xyz point: {num_xyz_point} class: {idx2class[obj_id]} {obj_id}")

                    bgr = mmcv.imread(im_path, "color")

                    print(f"xyz min {xyz.min()} max {xyz.max()}")
                    print(xyz_info["xyz_crop"].shape, xyz_info["xyxy"])
                    show_ims = [
                        bgr[:, :, [2, 1, 0]],
                        get_emb_show(xyz),
                        get_emb_show(xyz_info["xyz_crop"].astype("float32")),
                        mask,
                    ]

                    show_titles = ["color", "xyz", "xyz_crop", "mask"]
                    grid_show(show_ims, show_titles, row=2, col=2)
                    raise
                re, te = calc_rt_dist_m(pose_est, pose)
                r_errors.append(re)
                t_errors.append(te)
                if not (re < 5 and te < 0.05):
                    logger.warn(f"{xyz_path} re: {re}, te: {te} class: {idx2class[obj_id]} {obj_id} mask area: {area}")
        # stat results for this scene
        r_errors = np.array(r_errors)
        t_errors = np.array(t_errors)
        logger.info(f"r errors: min {r_errors.min()} max {r_errors.max()} mean {r_errors.mean()} std {r_errors.std()}")
        logger.info(f"t errors: min {t_errors.min()} max {t_errors.max()} mean {t_errors.mean()} std {t_errors.std()}")


if __name__ == "__main__":
    import argparse
    import time

    import setproctitle

    parser = argparse.ArgumentParser(description="verify lm_egl xyz")
    parser.add_argument("--gpu", type=str, default="0", help="gpu")
    parser.add_argument("--vis", default=False, action="store_true", help="vis")
    args = parser.parse_args()

    height = IM_H
    width = IM_W

    VIS = args.vis

    T_begin = time.perf_counter()
    setproctitle.setproctitle("verify xyz egl")
    xyz_gen = XyzVerify()
    xyz_gen.main()
    T_end = time.perf_counter() - T_begin
    print("total time: ", T_end)
