import os.path as osp
import sys
import numpy as np
import mmcv
from tqdm import tqdm
from functools import cmp_to_key

cur_dir = osp.dirname(osp.abspath(__file__))
PROJ_ROOT = osp.normpath(osp.join(cur_dir, "../../../../"))
sys.path.insert(0, PROJ_ROOT)
from lib.pysixd import inout, misc
from lib.utils.bbox_utils import xyxy_to_xywh
from lib.utils.utils import wprint, iprint

id2obj = {i: str(i) for i in range(1, 31)}
obj_num = len(id2obj)
obj2id = {_name: _id for _id, _name in id2obj.items()}


if __name__ == "__main__":
    # path for merged json
    det_path = osp.join(
        PROJ_ROOT,
        "datasets/BOP_DATASETS/tless/test/test_bboxes/",
        "yolov4x_tless_only_pbr_bop.json",
    )

    out_root = "output/gdrn/tless/resnest50d_online_AugCosyAAEGray_mlBCE_Tless_pbr_100e_bop_test"
    pred_paths = (
        "output/gdrn/tless/resnest50d_online_AugCosyAAEGray_mlBCE_Tless_pbr_100e_bop_test/all_tless-test_yolov4.csv"
    )
    obj_names = [obj for obj in obj2id]

    new_res_dict = {}
    for obj_name, pred_name in zip(obj_names, pred_paths):
        short_name = name_long2short(obj_name)
        assert short_name in pred_name, "{} not in {}".format(short_name, pred_name)

        pred_path = osp.join(out_root, pred_name)
        assert osp.exists(pred_path), pred_path
        iprint(obj_name, pred_path)

        # pkl  scene_im_id key, list of preds
        preds = mmcv.load(pred_path)

        for scene_im_id, pred_list in preds.items():
            for pred in pred_list:
                obj_id = pred["obj_id"]
                score = pred["score"]
                bbox_est = pred["bbox_est"]  # xyxy
                bbox_est_xywh = xyxy_to_xywh(bbox_est)

                R_est = pred["R"]
                t_est = pred["t"]
                pose_est = np.hstack([R_est, t_est.reshape(3, 1)])
                cur_new_res = {
                    "obj_id": obj_id,
                    "score": float(score),
                    "bbox_est": bbox_est_xywh.tolist(),
                    "pose_est": pose_est.tolist(),
                }
                if scene_im_id not in new_res_dict:
                    new_res_dict[scene_im_id] = []
                new_res_dict[scene_im_id].append(cur_new_res)

    def mycmp(x, y):
        # compare two scene_im_id
        x_scene_id = int(x[0].split("/")[0])
        y_scene_id = int(y[0].split("/")[0])
        if x_scene_id == y_scene_id:
            x_im_id = int(x[0].split("/")[1])
            y_im_id = int(y[0].split("/")[1])
            return x_im_id - y_im_id
        else:
            return x_scene_id - y_scene_id

    new_res_dict_sorted = dict(sorted(new_res_dict.items(), key=cmp_to_key(mycmp)))
    inout.save_json(new_res_path, new_res_dict_sorted)
    iprint()
    iprint("merged new result (json) path: {}".format(new_res_path))
