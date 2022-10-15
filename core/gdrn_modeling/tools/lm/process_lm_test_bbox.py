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
    mmcv.mkdir_or_exist(OUT_DIR)

    orig_data_path = osp.expanduser("~/Downloads/CDPN_LineMOD_Test.npy")
    data = np.load(orig_data_path, allow_pickle=True).item()
    print(data.keys())
    for obj in tqdm(ref.lm_full.objects):
        if obj in ["bowl", "cup"]:
            continue
        ori_obj = obj
        obj_id = ref.lm_full.obj2id[obj]

        result_dir = osp.join(OUT_DIR, f"{obj_id:06d}")
        mmcv.mkdir_or_exist(result_dir)
        faster_res_path = osp.join(result_dir, "bbox_faster.json")
        yolov3_res_path = osp.join(result_dir, "bbox_yolov3.json")
        tiny_yolov3_res_path = osp.join(result_dir, "bbox_tiny_yolov3.json")

        if obj == "benchvise":
            ori_obj = "benchviseblue"
        # print(data[ori_obj].keys())
        faster_res = {}
        yolov3_res = {}
        tiny_yolov3_res = {}
        for im_i, im_path in enumerate(tqdm(data[ori_obj]["imgPath"])):
            im_id = int(im_path.split("/")[-1].split("-")[0])  # 1-based
            bop_im_id = im_id - 1  # 0-based
            # print(data[ori_obj]['faster_bbox'][im_i])
            if str(bop_im_id) not in faster_res:
                faster_res[str(bop_im_id)] = []
            faster_res[str(bop_im_id)].append(
                {
                    "bbox": data[ori_obj]["faster_bbox"][im_i].tolist(),
                    "obj_id": obj_id,
                }
            )

            if str(bop_im_id) not in yolov3_res:
                yolov3_res[str(bop_im_id)] = []
            yolov3_res[str(bop_im_id)].append(
                {
                    "bbox": data[ori_obj]["yolov3_bbox"][im_i].tolist(),
                    "obj_id": obj_id,
                }
            )

            if str(bop_im_id) not in tiny_yolov3_res:
                tiny_yolov3_res[str(bop_im_id)] = []
            tiny_yolov3_res[str(bop_im_id)].append(
                {
                    "bbox": data[ori_obj]["tiny_yolov3_bbox"][im_i].tolist(),
                    "obj_id": obj_id,
                }
            )

        mmcv.dump(faster_res, faster_res_path)
        mmcv.dump(yolov3_res, yolov3_res_path)
        mmcv.dump(tiny_yolov3_res, tiny_yolov3_res_path)


if __name__ == "__main__":
    main()
