# compute fps (farthest point sampling) for models
import os.path as osp
import sys
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F

cur_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(cur_dir, "../../../../"))
import mmcv
from lib.pysixd import inout, misc
import ref
from core.utils.data_utils import get_fps_and_center


model_dir = ref.lm_full.model_dir
id2obj = ref.lm_full.id2obj


def main():
    vertex_scale = 0.001

    for obj_id in tqdm(id2obj):
        print(obj_id)
        model_path = osp.join(model_dir, f"obj_{obj_id:06d}.ply")
        model = inout.load_ply(model_path, vertex_scale=vertex_scale)
        # 4, 8, 12, 16, 20, 32, 64, 128, 256
        num_fps = 128
        print("num_fps: ", num_fps)
        fps_points_center = get_fps_and_center(model["pts"], num_fps=num_fps, init_center=True)

        fps_points_center_1 = get_fps_and_center(model["pts"], num_fps=num_fps, init_center=True)

        print("allclose: ", np.allclose(fps_points_center, fps_points_center_1))


if __name__ == "__main__":
    main()
