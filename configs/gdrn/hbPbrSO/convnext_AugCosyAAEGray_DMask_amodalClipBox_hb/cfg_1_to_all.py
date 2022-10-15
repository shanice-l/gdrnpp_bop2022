from mmcv import Config
import os.path as osp
import os
from tqdm import tqdm

cur_dir = osp.normpath(osp.dirname(osp.abspath(__file__)))

base_cfg_name = "01_bear.py"
base_obj_name = "01_bear"

# -----------------------------------------------------------------
id2obj = {
    1: "01_bear",
    3: "03_round_car",
    4: "04_thin_cow",
    8: "08_green_rabbit",
    9: "09_holepuncher",
    10: "10",
    12: "12",
    15: "15",
    17: "17",
    18: "18_jaffa_cakes_box",
    19: "19_minions",  # small yellow man
    22: "22_rhinoceros",  # xi niu
    23: "23_dog",
    29: "29_tea_box",
    32: "32_car",
    33: "33_yellow_rabbit",
}

obj2id = {_name: _id for _id, _name in id2obj.items()}


def main():
    base_cfg_path = osp.join(cur_dir, base_cfg_name)
    assert osp.exists(base_cfg_path), base_cfg_path  # make sure base cfg is in this dir
    cfg = Config.fromfile(base_cfg_path)

    for obj_id, obj_name in tqdm(id2obj.items()):
        print(obj_name)
        if obj_name == base_obj_name:
            continue
        # NOTE: what fields should be updated ---------------------------
        new_cfg_dict = dict(
            _base_="./{}".format(base_cfg_name),
            OUTPUT_DIR=cfg.OUTPUT_DIR.replace(base_obj_name, obj_name),
            DATASETS=dict(
                TRAIN=("hb_{}_train_pbr".format(obj_name),),
            ),
        )
        # ----------------------------------------------------------------------
        new_cfg_path = osp.join(cur_dir, base_cfg_name.replace(base_obj_name, obj_name))
        if osp.exists(new_cfg_path):
            raise RuntimeError("new cfg exists!")
        new_cfg = Config(new_cfg_dict)
        with open(new_cfg_path, "w") as f:
            f.write(new_cfg.pretty_text)

    # re-format
    os.system("black -l 120 {}".format(cur_dir))


if __name__ == "__main__":
    main()
