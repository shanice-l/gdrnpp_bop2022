import mmcv
import os.path as osp
import json

cur_dir = osp.dirname(osp.abspath(__file__))

# 33 objects
IDX2CLASS = {i: str(i) for i in range(1, 28 + 1)}
CLASSES = IDX2CLASS.values()
CLASSES = list(sorted(CLASSES))
CLASS2IDX = {cls_name: idx for idx, cls_name in IDX2CLASS.items()}

data_root = "datasets/BOP_DATASETS/itodd"


def main():
    val_targets = []  # {"im_id": , "inst_count": , "obj_id": , "scene_id": }
    val_scenes = list(range(1, 2))
    for scene_id in val_scenes:
        print("scene_id", scene_id)
        BOP_gt_file = osp.join(data_root, f"val/{scene_id:06d}/scene_gt.json")
        assert osp.exists(BOP_gt_file), BOP_gt_file
        gt_dict = mmcv.load(BOP_gt_file)
        all_ids = [int(k) for k in gt_dict.keys()]
        print(len(all_ids))
        for idx in all_ids:
            annos = gt_dict[str(idx)]
            obj_ids = [anno["obj_id"] for anno in annos]
            num_inst_dict = {}
            # stat num instances for each obj
            for obj_id in obj_ids:
                if obj_id not in num_inst_dict:
                    num_inst_dict[obj_id] = 1
                else:
                    num_inst_dict[obj_id] += 1
            for obj_id in num_inst_dict:
                target = {
                    "im_id": idx,
                    "inst_count": num_inst_dict[obj_id],
                    "obj_id": obj_id,
                    "scene_id": scene_id,
                }
                val_targets.append(target)
    res_file = osp.join(cur_dir, "itodd_val_targets_all.json")
    print(res_file)
    print(len(val_targets))  # 23120
    with open(res_file, "w") as f:
        f.write("[\n" + ",\n".join(json.dumps(item) for item in val_targets) + "]\n")
    print("done")


if __name__ == "__main__":
    main()
