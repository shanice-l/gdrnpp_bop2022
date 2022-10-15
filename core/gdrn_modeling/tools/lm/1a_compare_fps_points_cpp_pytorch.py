# compute fps (farthest point sampling) for models
import os.path as osp
import sys
from tqdm import tqdm
import torch
import torch.nn.functional as F

cur_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(cur_dir, "../../../../"))
import mmcv
from lib.pysixd import inout, misc
import ref
from core.utils.data_utils import get_fps_and_center

from core.utils.farthest_points_torch import (
    farthest_points,
    get_fps_and_center_torch,
)


model_dir = ref.lm_full.model_dir
id2obj = ref.lm_full.id2obj


def main():
    vertex_scale = 0.001

    for obj_id in tqdm(id2obj):
        print(obj_id)
        model_path = osp.join(model_dir, f"obj_{obj_id:06d}.ply")
        model = inout.load_ply(model_path, vertex_scale=vertex_scale)
        # 4, 8, 12, 16, 20, 32, 64, 256
        num_fps = 4
        print("num_fps: ", num_fps)
        fps_points_center = get_fps_and_center(model["pts"], num_fps=num_fps, init_center=True)
        points_th = torch.tensor(model["pts"], dtype=torch.float32)
        fps_points_center_th = get_fps_and_center_torch(points_th, num_fps=num_fps, init_center=True)
        print("fps_points_center: ", fps_points_center)
        print()
        print("fps_points_center_torch: ", fps_points_center_th)
        """The results are not identical"""


if __name__ == "__main__":
    main()
