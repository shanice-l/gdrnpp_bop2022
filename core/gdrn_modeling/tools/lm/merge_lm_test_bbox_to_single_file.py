import numpy as np

import os.path as osp
import sys
import os

cur_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(cur_dir, "../../../.."))
import ref
import mmcv
from tqdm import tqdm


def main():
    OUT_DIR = "datasets/BOP_DATASETS/lm/test/test_bboxes/"
    out_faster_path = osp.join(OUT_DIR, "bbox_faster_all.json")
    out_yolov3_path = osp.join(OUT_DIR, "bbox_yolov3_all.json")
    out_tiny_yolov3_path = osp.join(OUT_DIR, "bbox_tiny_yolov3_all.json")

    faster_res = {}
    yolov3_res = {}
    tiny_yolov3_res = {}
    for obj in tqdm(ref.lm_full.objects):
        if obj in ["bowl", "cup"]:
            continue
        obj_id = ref.lm_full.obj2id[obj]
        scene_id = obj_id

        result_dir = osp.join(OUT_DIR, f"{obj_id:06d}")
        faster_res_path = osp.join(result_dir, "bbox_faster.json")
        yolov3_res_path = osp.join(result_dir, "bbox_yolov3.json")
        tiny_yolov3_res_path = osp.join(result_dir, "bbox_tiny_yolov3.json")

        cur_faster_res = mmcv.load(faster_res_path)
        cur_yolov3_res = mmcv.load(yolov3_res_path)
        cur_tiny_yolov3_res = mmcv.load(tiny_yolov3_res_path)

        for im_id in cur_faster_res:
            int_im_id = int(im_id)
            scene_im_id = "{}/{}".format(scene_id, im_id)
            dets = cur_faster_res[im_id]

            faster_res[scene_im_id] = []
            for det in dets:
                inst = {
                    "obj_id": det["obj_id"],
                    "bbox_est": det["bbox"],
                    "score": 1,
                }
                faster_res[scene_im_id].append(inst)

        for im_id in cur_yolov3_res:
            int_im_id = int(im_id)
            scene_im_id = "{}/{}".format(scene_id, im_id)
            dets = cur_yolov3_res[im_id]

            yolov3_res[scene_im_id] = []
            for det in dets:
                inst = {
                    "obj_id": det["obj_id"],
                    "bbox_est": det["bbox"],
                    "score": 1,
                }
                yolov3_res[scene_im_id].append(inst)

        for im_id in cur_tiny_yolov3_res:
            int_im_id = int(im_id)
            scene_im_id = "{}/{}".format(scene_id, int_im_id)
            dets = cur_tiny_yolov3_res[im_id]

            tiny_yolov3_res[scene_im_id] = []
            for det in dets:
                inst = {
                    "obj_id": det["obj_id"],
                    "bbox_est": det["bbox"],
                    "score": 1,
                }
                tiny_yolov3_res[scene_im_id].append(inst)

    mmcv.dump(faster_res, out_faster_path)
    mmcv.dump(yolov3_res, out_yolov3_path)
    mmcv.dump(tiny_yolov3_res, out_tiny_yolov3_path)


if __name__ == "__main__":
    main()
