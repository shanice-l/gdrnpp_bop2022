import os.path as osp
import sys
import numpy as np
import mmcv
from tqdm import tqdm

cur_dir = osp.dirname(osp.abspath(__file__))
PROJ_ROOT = osp.normpath(osp.join(cur_dir, "../../../../"))
sys.path.insert(0, PROJ_ROOT)
from lib.pysixd import inout, misc
from lib.utils.bbox_utils import xyxy_to_xywh


id2obj = {
    1: "ape",
    #  2: 'benchvise',
    #  3: 'bowl',
    #  4: 'camera',
    5: "can",
    6: "cat",
    #  7: 'cup',
    8: "driller",
    9: "duck",
    10: "eggbox",
    11: "glue",
    12: "holepuncher",
    #  13: 'iron',
    #  14: 'lamp',
    #  15: 'phone'
}
obj_num = len(id2obj)
obj2id = {_name: _id for _id, _name in id2obj.items()}


if __name__ == "__main__":
    out_root = "output/gdrn/lmoPbrSO/resnest50d_online_AugCosyAAEGray_mlBCE_DoubleMask_lmo_pbr_100e/"
    pred_paths = [
        "duck/inference_model_final_wo_optim-3cc3dbe6/duck_frames/results.pkl",
    ]
    obj_names = ["duck"]

    new_res_dict = {}
    for obj_name, pred_name in zip(obj_names, pred_paths):
        assert obj_name in pred_name, "{} not in {}".format(obj_name, pred_name)

        pred_path = osp.join(out_root, pred_name)
        assert osp.exists(pred_path), pred_path
        print(obj_name, pred_path)

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

    new_res_path = osp.join(out_root, "duck/inference_model_final_wo_optim-3cc3dbe6/duck_frames/gdrn_results.json")
    inout.save_json(new_res_path, new_res_dict)
    print()
    print("new result path: {}".format(new_res_path))
