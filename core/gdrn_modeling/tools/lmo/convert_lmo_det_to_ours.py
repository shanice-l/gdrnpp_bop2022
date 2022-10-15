from matplotlib import category
import mmcv
import sys
import argparse
import json

parser = argparse.ArgumentParser(description="convert det from bop format to ours")
parser.add_argument("--ipath", type=str, default="0", help="input path")
parser.add_argument("--opath", type=str, default="0", help="outpur path")
args = parser.parse_args()

ds = mmcv.load(args.ipath)

outs = {}

catid2obj = {
    1: "ape",
    5: "can",
    6: "cat",
    8: "driller",
    9: "duck",
    10: "eggbox",
    11: "glue",
    12: "holepuncher",
}
objects = [
    "ape",
    "can",
    "cat",
    "driller",
    "duck",
    "eggbox",
    "glue",
    "holepuncher",
]
obj2id = {_name: _id for _id, _name in catid2obj.items()}


for d in ds:
    scene_id = d["scene_id"]
    image_id = d["image_id"]
    scene_im_id = f"{scene_id}/{image_id}"

    obj_id = d["category_id"]
    obj_name = objects[obj_id - 1]
    category_id = obj2id[obj_name]

    score = d["score"]

    bbox = d["bbox"]
    time = d["time"]

    cur_dict = {
        "bbox_est": bbox,
        "obj_id": category_id,
        "score": score,
        "time": time,
    }

    if scene_im_id in outs.keys():
        outs[scene_im_id].append(cur_dict)
    else:
        outs[scene_im_id] = [cur_dict]


def save_json(path, content, sort=False):
    """Saves the provided content to a JSON file.

    :param path: Path to the output JSON file.
    :param content: Dictionary/list to save.
    """
    with open(path, "w") as f:

        if isinstance(content, dict):
            f.write("{\n")
            if sort:
                content_sorted = sorted(content.items(), key=lambda x: x[0])
            else:
                content_sorted = content.items()
            for elem_id, (k, v) in enumerate(content_sorted):
                f.write('  "{}": {}'.format(k, json.dumps(v, sort_keys=True)))
                if elem_id != len(content) - 1:
                    f.write(",")
                f.write("\n")
            f.write("}")

        elif isinstance(content, list):
            f.write("[\n")
            for elem_id, elem in enumerate(content):
                f.write("  {}".format(json.dumps(elem, sort_keys=True)))
                if elem_id != len(content) - 1:
                    f.write(",")
                f.write("\n")
            f.write("]")

        else:
            json.dump(content, f, sort_keys=True)


save_json(args.opath, outs)
