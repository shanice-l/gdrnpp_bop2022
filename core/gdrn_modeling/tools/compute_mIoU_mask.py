import argparse
import os
import os.path as osp
import sys
from tqdm import tqdm
import numpy as np
import mmcv
from detectron2.data import DatasetCatalog, MetadataCatalog

cur_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(cur_dir, "../../../"))

from core.gdrn_modeling.datasets.dataset_factory import register_datasets
from lib.utils.mask_utils import cocosegm2mask

import ref


"""
python core/gdrn_modeling/tools/compute_mIoU_mask.py --dataset lm_real_ape_test --cls ape  \
    --res_path output/gdrn/lm_pbr/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_lm_pbr_100e/ape/inference_model_final_wo_optim-e8c99c96/lm_real_ape_test/results.pkl
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Compute f1 score of mask.")
    parser.add_argument("--dataset", type=str, help="dataset_name")
    parser.add_argument("--cls", type=str, help="class name")
    parser.add_argument("--mask_type", type=str, default="vis", help="mask type: vis | full")
    parser.add_argument("--res_path", type=str, help="path to results.")
    parser.add_argument("--height", type=int, default=480, help="im height")
    parser.add_argument("--width", type=int, default=640, help="im width")
    args = parser.parse_args()
    return args


args = parse_args()
HEIGHT = args.height
WIDTH = args.width

if args.mask_type == "vis":
    print("evaluating visible mask")
elif args.mask_type == "full":
    print("evaluating full mask")
else:
    raise ValueError("Unknown mask type: {}, available options: vis | full".format(args.mask_type))


def compute_mean_iou(pred, label):
    # https://github.com/GeorgeSeif/Semantic-Segmentation-Suite/blob/master/utils/utils.py
    unique_labels = np.unique(label)
    num_unique_labels = len(unique_labels)

    I = np.zeros(num_unique_labels)
    U = np.zeros(num_unique_labels)

    for index, val in enumerate(unique_labels):
        pred_i = pred == val
        label_i = label == val

        I[index] = float(np.sum(np.logical_and(label_i, pred_i)))
        U[index] = float(np.sum(np.logical_or(label_i, pred_i)))

    mean_iou = np.mean(I / U)
    return mean_iou


def eval_mIoU():
    # load gt mask
    # load pred mask
    # compute score
    dset_name = args.dataset
    res_path = args.res_path
    cls_name = args.cls
    assert osp.exists(res_path), res_path
    assert cls_name in dset_name, f"{cls_name} not in {dset_name}"
    assert cls_name in res_path, f"{cls_name} not in {res_path}"
    print("dataset: ", dset_name)
    print("class: ", cls_name)
    print("results path: ", res_path)
    register_datasets([dset_name])
    dataset_dicts = DatasetCatalog.get(dset_name)
    dset_meta = MetadataCatalog.get(dset_name)
    data_ref = ref.__dict__[dset_meta.ref_key]
    objs = dset_meta.objs
    obj_id = data_ref.obj2id[cls_name]

    # predictions
    init_res = mmcv.load(res_path)

    mIoUs = []
    num_detected = 0
    for dic in tqdm(dataset_dicts):
        scene_im_id = dic["scene_im_id"]
        if scene_im_id not in init_res:
            mIoUs.append(0.0)
            continue

        cur_res_list = init_res[scene_im_id]
        pred_obj_ids = [_res["obj_id"] for _res in cur_res_list]

        # load pred mask
        if obj_id not in pred_obj_ids:
            mIoUs.append(0.0)
            continue

        num_detected += 1
        pred_ind = pred_obj_ids.index(obj_id)  # NOTE: only works for datasets with only 1 instance per object

        if args.mask_type == "vis":
            pred_mask_rle = cur_res_list[pred_ind]["mask"]
        elif args.mask_type == "full":
            pred_mask_rle = cur_res_list[pred_ind]["full_mask"]
        else:
            raise ValueError("Unknonw mask type: {}".format(args.mask_type))

        pred_mask = cocosegm2mask(pred_mask_rle, h=HEIGHT, w=WIDTH)
        # load gt mask
        annos = dic["annotations"]
        for anno_i, anno in enumerate(annos):
            obj_name_i = objs[anno["category_id"]]
            if obj_name_i == cls_name:
                if args.mask_type == "vis":
                    gt_rle = anno["segmentation"]
                elif args.mask_type == "full":
                    gt_rle = anno["mask_full"]
                else:
                    raise ValueError("Unknonw mask type: {}".format(args.mask_type))
                gt_mask = cocosegm2mask(gt_rle, h=HEIGHT, w=WIDTH)
                break
        cur_mIoU = compute_mean_iou(pred_mask.flatten(), gt_mask.flatten())
        mIoUs.append(cur_mIoU)

    print("mask type: {}".format(args.mask_type))
    print("dataset name: ", dset_name, "object: ", cls_name)
    print("dataset length: ", len(dataset_dicts))
    print("length of mIoUs: ", len(mIoUs))
    print("num detected: ", num_detected)
    print(f"mIoU (%): {np.mean(mIoUs) * 100:.2f}")


if __name__ == "__main__":
    eval_mIoU()
