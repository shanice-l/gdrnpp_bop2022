import argparse
import os
import os.path as osp
import sys
from tqdm import tqdm
import torch
import mmcv
from detectron2.data import DatasetCatalog, MetadataCatalog
cur_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(cur_dir, "../../../"))
from det.yolox.data.datasets.dataset_factory import register_datasets
from lib.utils.utils import iprint, dprint
import ref


def get_parser():
    parser = argparse.ArgumentParser(description="Convert results to bop format")
    parser.add_argument(
        "--path",
        default="output/instances_predictions.pth",
        help="path to prediction results",
    )
    parser.add_argument(
        "--dataset",
        default="lmo_bop_test",
        help="registered dataset name",
    )
    return parser


def main():
    args = get_parser().parse_args()
    dset_name = args.dataset
    dprint("dataset: ", dset_name)
    register_datasets([dset_name])

    meta = MetadataCatalog.get(dset_name)
    objs = meta.objs
    ref_key = meta.ref_key
    data_ref = ref.__dict__[ref_key]

    dicts = DatasetCatalog.get(dset_name)

    dprint("results load from: ", args.path)
    results = torch.load(args.path)

    bop_results = {}
    for i, res in enumerate(tqdm(results)):
        image_id = res['image_id']  # not bop im id
        # get gts
        for dic in dicts:
            if dic['image_id'] == image_id:
                break
        scene_im_id = dic['scene_im_id']
        # scene_im_id_split = scene_im_id.split("/")
        # scene_id = int(scene_im_id_split[0])
        # im_id = int(scene_im_id_split[1])

        pred_instances = res['instances']
        if "time" in res:
            time = res['time']
        else:
            time = 0
        res_insts = []
        for inst in pred_instances:
            label = inst["category_id"]
            if label >= len(objs):
                dprint(f"label: {label} is not valid. num objs: {len(objs)}")
                continue
            obj_name = objs[label]

            obj_id = data_ref.obj2id[obj_name]

            bbox = inst['bbox']
            score = inst['score']
            res_inst = {
                "obj_id": obj_id,
                "bbox_est": bbox,  # xywh format
                "score": score,
                "time": time
            }
            res_insts.append(res_inst)

        bop_results[scene_im_id] = res_insts

    # dump results
    result_name = "instances_predictions_bop.json"
    result_path = osp.join(osp.dirname(args.path), result_name)
    mmcv.dump(bop_results, result_path)
    dprint(f"results were written to {result_path}")


if __name__ == "__main__":
    main()
