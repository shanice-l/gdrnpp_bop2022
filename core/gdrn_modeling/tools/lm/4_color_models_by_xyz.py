"""colorize models by xyz coordinates."""
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


model_dir = ref.lm_full.model_dir
id2obj = ref.lm_full.id2obj


def main():
    print("generate models_xyz")
    for obj_id in id2obj:
        model_path = osp.join(model_dir, f"obj_{obj_id:06d}.ply")
        model = inout.load_ply(model_path, vertex_scale=0.001)

        out_model_dir = osp.join(ref.lm_full.dataset_root, "models_xyz")
        mmcv.mkdir_or_exist(out_model_dir)

        model_xyz = copy.deepcopy(model)

        pts = model["pts"]
        # normalize to [0,1]
        pts = (pts - pts.min(0)[None]) / (pts.max(0)[None] - pts.min(0)[None])
        model_xyz["colors"] = (pts * 255).astype("uint8")

        save_model_path = osp.join(out_model_dir, f"obj_{obj_id:06d}.ply")
        inout.save_ply(save_model_path, model_xyz)


if __name__ == "__main__":
    main()
