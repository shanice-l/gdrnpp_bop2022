# encoding: utf-8
"""This file includes necessary params, info."""
import os.path as osp

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
# HB DATASET
# ---------------------------------------------------------------- #
dataset_root = osp.join(bop_root, "hb")
train_dir = osp.join(dataset_root, "train")
test_dir = osp.join(dataset_root, "test")
model_dir = osp.join(dataset_root, "models")
vertex_scale = 0.001
model_eval_dir = osp.join(dataset_root, "models_eval")

# object info
# id2obj = {idx: str(idx) for idx in [1, 3, 4, 8, 9, 10, 12, 15, 17, 18, 19, 22, 23, 29, 32, 33]
#          }  # only 16 classes are selected
id2obj = {
    1: "01_bear",
    # 2: "02_benchvise",
    3: "03_round_car",
    4: "04_thin_cow",
    # 5: "05_fat_cow",
    # 6: "06_mug",
    # 7: "07_driller",
    8: "08_green_rabbit",
    9: "09_holepuncher",
    10: "10",
    # 11: "11",
    12: "12",
    # 13: "13",
    # 14: "14",
    15: "15",
    # 16: "16",
    17: "17",
    18: "18_jaffa_cakes_box",
    19: "19_minions",  # small yellow man
    # 20: "20_color_dog",
    # 21: "21_phone",
    22: "22_rhinoceros",  # xi niu
    23: "23_dog",
    # 24: "24",
    # 25: "25_car",
    # 26: "26_motorcycle",
    # 27: "27_high_heels",
    # 28: "28_stegosaurus",   # jian chi long
    29: "29_tea_box",
    # 30: "30_triceratops",  # san jiao long
    # 31: "31_toy_baby",
    32: "32_car",
    33: "33_yellow_rabbit",
}
objects = [str(obj) for obj in id2obj.values()]
obj_num = len(id2obj)
obj2id = {cls_name: cls_idx for cls_idx, cls_name in id2obj.items()}

model_paths = [osp.join(model_dir, "obj_{:06d}.ply").format(_id) for _id in id2obj]
texture_paths = None
model_colors = [((i + 1) * 10, (i + 1) * 10, (i + 1) * 10) for i in range(obj_num)]  # for renderer

# Camera info
width = 640
height = 480
zNear = 0.25
zFar = 6.0
center = (height / 2, width / 2)
camera_matrix = np.array([[537.4799, 0.0, 318.8965], [0.0, 536.1447, 238.3781], [0.0, 0.0, 1.0]])
