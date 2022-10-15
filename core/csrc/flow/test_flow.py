"""torch version flow is faster when inputs and outputs are cuda tensors."""
from __future__ import absolute_import, division, print_function
import sys
import os
import os.path as osp
import numpy as np
import torch
import cv2
import mmcv
from time import time
import matplotlib.pyplot as plt

cur_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, cur_dir)
sys.path.insert(0, os.path.join(cur_dir, "../../.."))

from flow_torch import flow as flow_cuda

from lib.pysixd import RT_transform
from lib.vis_utils.image import grid_show
from lib.vis_utils.optflow import flow2rgb


CAMERA_MATRIX = np.array([[572.4114, 0, 325.2611], [0, 573.57043, 242.04899], [0, 0, 1]])
IDX2CLASS = {
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
CLASSES = IDX2CLASS.values()
CLASSES = sorted(CLASSES)
CLASS2IDX = {cls_name: idx for idx, cls_name in IDX2CLASS.items()}


if __name__ == "__main__":
    K = np.array([[572.4114, 0, 325.2611], [0, 573.57043, 242.04899], [0, 0, 1]])
    Kinv = np.linalg.inv(K)
    DEPTH_FACTOR = 1000.0
    flow_thresh = 3e-3
    batch_size = 8
    height = 480
    width = 640
    wh_rep = False
    device = "cuda:0"
    src_img_idx = [x * 100 + 1 for x in range(batch_size)]
    tgt_img_idx = [x * 100 + 31 for x in range(batch_size)]
    class_name = "driller"
    cls_idx = 8
    data_root = osp.join(cur_dir, "../../../datasets/")

    data_dir = osp.join(data_root, "BOP_DATASETS/lm/")
    model_dir = osp.join(data_dir, "models")
    color_path = osp.join(data_dir, "test/{:06d}/rgb/{:06d}.png")
    depth_path = osp.join(data_dir, "test/{:06d}/depth/{:06d}.png")
    gt_path = osp.join(data_dir, "test/{:06d}/scene_gt.json").format(cls_idx)
    gt_dict = mmcv.load(gt_path)

    def pose_at_i(index, gt_dict, cls_idx):
        instances = gt_dict[str(int(index))]
        for instance in instances:
            if instance["obj_id"] == cls_idx:
                R = np.array(instance["cam_R_m2c"]).reshape((3, 3))
                t = np.array(instance["cam_t_m2c"]).reshape((3,)) / 1000.0
                pose = np.zeros((3, 4))
                pose[:3, :3] = R
                pose[:3, 3] = t
                return pose
        return None

    # prepare input data
    v_depth_src = np.zeros((batch_size, 1, height, width), dtype=np.float32)
    v_depth_tgt = np.zeros((batch_size, 1, height, width), dtype=np.float32)
    v_pose_src = np.zeros((batch_size, 3, 4), dtype=np.float32)
    v_pose_tgt = np.zeros((batch_size, 3, 4), dtype=np.float32)
    KT_array = np.zeros((batch_size, 3, 4), dtype=np.float32)
    Kinv_array = np.zeros((batch_size, 3, 3), dtype=np.float32)

    for i in range(batch_size):
        depth_src_path = depth_path.format(cls_idx, src_img_idx[i])
        depth_tgt_path = depth_path.format(cls_idx, tgt_img_idx[i])
        assert osp.exists(depth_src_path), "no {}".format(depth_src_path)
        assert osp.exists(depth_tgt_path), "no {}".format(depth_tgt_path)
        v_depth_src[i, 0, :, :] = cv2.imread(depth_src_path, cv2.IMREAD_UNCHANGED) / DEPTH_FACTOR
        v_depth_tgt[i, 0, :, :] = cv2.imread(depth_tgt_path, cv2.IMREAD_UNCHANGED) / DEPTH_FACTOR
        # import pdb; pdb.set_trace()
        v_pose_src[i] = pose_at_i(src_img_idx[i], gt_dict, cls_idx)
        v_pose_tgt[i] = pose_at_i(tgt_img_idx[i], gt_dict, cls_idx)

        se3_m = np.zeros([3, 4])
        se3_rotm, se3_t = RT_transform.calc_se3(v_pose_src[i], v_pose_tgt[i])
        se3_m[:, :3] = se3_rotm
        se3_m[:, 3] = se3_t
        KT_array[i] = np.dot(K, se3_m)
        Kinv_array[i] = Kinv

    def to_tensor(_data):
        return torch.tensor(_data, dtype=torch.float32, device=device)

    flow_version = "torch"  # 'torch'
    depth_src_tensor = to_tensor(v_depth_src)
    depth_tgt_tensor = to_tensor(v_depth_tgt)
    pose_src_tensor = to_tensor(v_pose_src)
    pose_tgt_tensor = to_tensor(v_pose_tgt)
    K_batch = to_tensor(K).repeat(batch_size, 1, 1)

    print("start")
    """
    torch to cpu 0.0028679728507995605 s
    torch 2.4566650390625e-05 s
    """

    tic = time()
    for i in range(100):
        flow_all_torch, flow_weights_all_torch = flow_cuda(
            depth_src_tensor,
            depth_tgt_tensor,
            pose_src_tensor,
            pose_tgt_tensor,
            K_batch,
        )
        flow_all_torch = flow_all_torch.cpu().numpy()
        flow_weights_all_torch = flow_weights_all_torch.cpu().numpy()
    print("torch to cpu {} s".format((time() - tic) / 100))

    tic = time()
    for i in range(100):
        flow_all_torch, flow_weights_all_torch = flow_cuda(
            depth_src_tensor,
            depth_tgt_tensor,
            pose_src_tensor,
            pose_tgt_tensor,
            K_batch,
        )
    print("torch {} s".format((time() - tic) / 100))
    flow_all_torch = flow_all_torch.cpu().numpy()
    flow_weights_all_torch = flow_weights_all_torch.cpu().numpy()

    print("torch")
    print(
        "flow, shape: ",
        flow_all_torch.shape,
        "unique: ",
        np.unique(flow_all_torch),
    )
    print(
        "flow weights, shape: ",
        flow_weights_all_torch.shape,
        "unique: ",
        np.unique(flow_weights_all_torch),
    )

    for j in range(batch_size):
        img_src_path = color_path.format(cls_idx, src_img_idx[j])
        assert osp.exists(img_src_path), "no {}".format(img_src_path)
        img_tgt_path = color_path.format(cls_idx, tgt_img_idx[j])
        assert osp.exists(img_tgt_path)
        img_src = cv2.imread(img_src_path, cv2.IMREAD_COLOR)
        img_src = img_src[:, :, [2, 1, 0]]
        img_tgt = cv2.imread(img_tgt_path, cv2.IMREAD_COLOR)
        img_tgt = img_tgt[:, :, [2, 1, 0]]

        print("torch flow_all(unique): \n", np.unique(flow_all_torch[j]))
        flow_torch = np.squeeze(flow_all_torch[j, :, :, :].transpose((1, 2, 0)))
        flow_weights_torch = flow_weights_all_torch[j, 0, :, :].reshape((height, width, 1))

        flow_valid_torch = flow_torch * flow_weights_torch

        flow_show_mmcv_torch = flow2rgb(flow_valid_torch)

        depth_src = (
            cv2.imread(
                depth_path.format(cls_idx, src_img_idx[j]),
                cv2.IMREAD_UNCHANGED,
            )
            / DEPTH_FACTOR
        )
        depth_tgt = (
            cv2.imread(
                depth_path.format(cls_idx, tgt_img_idx[j]),
                cv2.IMREAD_UNCHANGED,
            )
            / DEPTH_FACTOR
        )

        show_imgs = [
            img_src,
            img_tgt,
            depth_src,
            depth_tgt,
            flow_show_mmcv_torch,
        ]  # , flow_show_sintel]
        show_titles = [
            "img_src",
            "img_tgt",
            "depth_src",
            "depth_tgt",
            "flow_show_mmcv_torch",
        ]  # , 'flow_show_sintel']

        grid_show(show_imgs, show_titles, row=3, col=3)
