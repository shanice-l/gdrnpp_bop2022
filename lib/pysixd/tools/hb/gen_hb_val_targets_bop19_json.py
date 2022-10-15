import mmcv
import os.path as osp
import json

cur_dir = osp.dirname(osp.abspath(__file__))

# 16 objects
IDX2CLASS = {
    1: "01_bear",
    # 2: "02_benchvise",
    3: "03_round_car",
    4: "04_thin_cow",
    # 5: "05_fat_cow",
    # 6: "06_mug",
    # 7: "07_driller",
    8: "08_green_rabbit",
    9: "09_holepuncher",
    10: "10",
    # 11: "11",
    12: "12",
    # 13: "13",
    # 14: "14",
    15: "15",
    # 16: "16",
    17: "17",
    18: "18_jaffa_cakes_box",
    19: "19_minions",  # small yellow man
    # 20: "20_color_dog",
    # 21: "21_phone",
    22: "22_rhinoceros",  # xi niu
    23: "23_dog",
    # 24: "24",
    # 25: "25_car",
    # 26: "26_motorcycle",
    # 27: "27_high_heels",
    # 28: "28_stegosaurus",   # jian chi long
    29: "29_tea_box",
    # 30: "30_triceratops",  # san jiao long
    # 31: "31_toy_baby",
    32: "32_car",
    33: "33_yellow_rabbit",
}
CLASSES = IDX2CLASS.values()
CLASSES = list(sorted(CLASSES))
CLASS2IDX = {cls_name: idx for idx, cls_name in IDX2CLASS.items()}

data_root = "datasets/BOP_DATASETS/hb"


def main():
    val_targets = []  # {"im_id": , "inst_count": , "obj_id": , "scene_id": }
    val_scenes = [3, 5, 13]
    for scene_id in val_scenes:
        print("scene_id", scene_id)
        BOP_gt_file = osp.join(data_root, f"val_primesense/{scene_id:06d}/scene_gt.json")
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
                if obj_id not in IDX2CLASS:
                    continue
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
    res_file = osp.join(cur_dir, "hb_val_targets_bop19.json")
    print(res_file)
    print(len(val_targets))  # 5780
    with open(res_file, "w") as f:
        f.write("[\n" + ",\n".join(json.dumps(item) for item in val_targets) + "]\n")
    print("done")


if __name__ == "__main__":
    main()
