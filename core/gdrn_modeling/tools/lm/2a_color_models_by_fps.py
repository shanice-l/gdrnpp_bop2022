"""colorize models by fps points."""
import os.path as osp
import sys
import numpy as np
from tqdm import tqdm
import copy
from scipy.spatial.distance import cdist

cur_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(cur_dir, "../../../../"))
import mmcv
from lib.pysixd import inout, misc
import ref
from core.utils.data_utils import get_fps_and_center


model_dir = ref.lm_full.model_dir
id2obj = ref.lm_full.id2obj


def region_id_to_color(region_ids, num_fps):
    if num_fps <= 24:
        region_ids = (region_ids + 1) * 10
    elif num_fps > 24 and num_fps <= 50:
        region_ids = (region_ids + 1) * 5
    elif num_fps > 50 and num_fps <= 84:
        region_ids = (region_ids + 1) * 3
    elif num_fps > 84 and num_fps <= 126:
        region_ids = (region_ids + 1) * 2
    elif num_fps > 126 and num_fps <= 254:
        region_ids = region_ids + 1
    else:
        pass
        # region_ids = region_ids
    return region_ids


def main():
    # load original models
    models = {}
    for obj_id in id2obj:
        model_path = osp.join(model_dir, f"obj_{obj_id:06d}.ply")
        model = inout.load_ply(model_path, vertex_scale=0.001)
        models[obj_id] = model

    for num_fps in tqdm([4, 8, 12, 16, 20, 32, 64, 256]):
        print("generate models_fps{}".format(num_fps))
        out_model_dir = osp.join(ref.lm_full.dataset_root, f"models_fps{num_fps}")
        mmcv.mkdir_or_exist(out_model_dir)

        fps_dict = mmcv.load(osp.join(model_dir, "fps_points.pkl"))

        for obj_id in tqdm(id2obj):
            model_fps = copy.deepcopy(models[obj_id])
            points = model_fps["pts"]  # nx3
            fps_points = fps_dict[str(obj_id)][f"fps{num_fps}_and_center"][:-1]  # fx3
            # idx: 0~num_fps-1
            # compute distance
            dists = cdist(points, fps_points)  # nxf
            region_ids = np.argmin(dists, axis=1)  # assign region label with the min dist
            region_ids = region_id_to_color(region_ids, num_fps)

            num_points = points.shape[0]
            colors = np.zeros((num_points, 3), dtype=np.uint8)  # int colors
            colors[:, 0] = region_ids

            model_fps["colors"] = colors

            save_model_path = osp.join(out_model_dir, f"obj_{obj_id:06d}.ply")
            inout.save_ply(save_model_path, model_fps)


if __name__ == "__main__":
    main()
