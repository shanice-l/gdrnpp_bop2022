# encoding: utf-8
"""This file includes necessary params, info."""
import os
import os.path as osp
import mmcv
import numpy as np

# ---------------------------------------------------------------- #
# ROOT PATH INFO
# ---------------------------------------------------------------- #
cur_dir = osp.abspath(osp.dirname(__file__))
root_dir = osp.normpath(osp.join(cur_dir, ".."))
output_dir = osp.join(root_dir, "output")  # directory storing experiment data (result, model checkpoints, etc).

data_root = osp.join(root_dir, "datasets")
bop_root = osp.join(data_root, "BOP_DATASETS/")
# ---------------------------------------------------------------- #
# TLESS DATASET
# ---------------------------------------------------------------- #
dataset_root = osp.join(bop_root, "tless")
train_real_dir = osp.join(dataset_root, "train_primesense")
train_render_dir = osp.join(dataset_root, "train_render_reconst")
test_dir = osp.join(dataset_root, "test_primesense")

# model_dir = osp.join(dataset_root, "models_reconst")  # use recon models as default
model_dir = osp.join(dataset_root, "models_cad")
model_cad = osp.join(dataset_root, "models_cad")
model_reconst_dir = osp.join(dataset_root, "models_reconst")
model_eval_dir = osp.join(dataset_root, "models_eval")
vertex_scale = 0.001
# object info
objects = [str(i) for i in range(1, 31)]
id2obj = {i: str(i) for i in range(1, 31)}

obj_num = len(id2obj)
obj2id = {_name: _id for _id, _name in id2obj.items()}

model_paths = [osp.join(model_dir, "obj_{:06d}.ply").format(_id) for _id in id2obj]
texture_paths = None
model_colors = [((i + 1) * 5, (i + 1) * 5, (i + 1) * 5) for i in range(obj_num)]  # for renderer

# Camera info
tr_real_width = 400
tr_real_height = 400
tr_render_width = 1280
tr_render_height = 1024
width = te_width = 720  # pbr size
height = te_height = 540  # pbr size
zNear = 0.25
zFar = 6.0
tr_real_center = (tr_real_height / 2, tr_real_width / 2)
tr_render_center = (tr_render_height / 2, tr_render_width / 2)
te_center = (te_width / 2.0, te_height / 2.0)
zNear = 0.25
zFar = 6.0

# NOTE: for tless, the camera matrix is not fixed!
camera_matrix = np.array([1075.65091572, 0.0, 360.0, 0.0, 1073.90347929, 270.0, 0.0, 0.0, 1.0]).reshape(3, 3)


diameters = (
    np.array(
        [
            63.5151,
            66.1512,
            65.3491,
            80.7257,
            108.69,
            108.265,
            178.615,
            217.156,
            144.546,
            90.2112,
            76.5978,
            86.0109,
            58.1257,
            71.9471,
            68.5692,
            69.1883,
            112.839,
            110.982,
            89.0689,
            98.8887,
            92.2527,
            92.2527,
            142.587,
            84.736,
            108.801,
            108.801,
            152.495,
            124.778,
            134.227,
            88.7538,
        ]
    )
    / 1000.0
)


def get_models_info():
    """key is str(obj_id)"""
    models_info_path = osp.join(model_dir, "models_info.json")
    assert osp.exists(models_info_path), models_info_path
    models_info = mmcv.load(models_info_path)  # key is str(obj_id)
    return models_info


# ref core/gdrn_modeling/tools/tless/tless_1_compute_fps.py
def get_fps_points():
    fps_points_path = osp.join(model_dir, "fps_points.pkl")
    assert osp.exists(fps_points_path), fps_points_path
    fps_dict = mmcv.load(fps_points_path)
    return fps_dict


# ref core/gdrn_modeling/tools/tless/tless_1_compute_keypoints_3d.py
def get_keypoints_3d():
    keypoints_3d_path = osp.join(model_dir, "keypoints_3d.pkl")
    assert osp.exists(keypoints_3d_path), keypoints_3d_path
    kpts_dict = mmcv.load(keypoints_3d_path)
    return kpts_dict
