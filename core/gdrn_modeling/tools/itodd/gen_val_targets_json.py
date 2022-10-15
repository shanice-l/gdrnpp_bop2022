import mmcv
import os.path as osp
import sys
from tqdm import tqdm

cur_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(cur_dir, "../../../../"))

from lib.pysixd import inout

path = "datasets/BOP_DATASETS/itodd/val/000001/scene_gt.json"
out_file = "datasets/BOP_DATASETS/itodd/val_targets.json"
targets = []

gt_dicts = mmcv.load(path)

for im_id, annos in gt_dicts.items():
    inst_count = len(annos)
    for anno in annos:
        obj_id = anno["obj_id"]
        d = {
            "im_id": int(im_id),
            "inst_count": inst_count,
            "obj_id": obj_id,
            "scene_id": 1,
        }
        targets.append(d)

# save to json
inout.save_json(out_file, targets)
