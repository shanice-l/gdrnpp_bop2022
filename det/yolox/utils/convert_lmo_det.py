from matplotlib import category
import mmcv
import sys
import argparse
import json
import copy

from detectron2.utils.file_io import PathManager

parser = argparse.ArgumentParser(description="convert lmo det from lm category to lmo category")
parser.add_argument("--input_path", type=str, default="0", help="input path")
parser.add_argument("--out_path", type=str, default="0", help="outpur path")
args = parser.parse_args()

ds = mmcv.load(args.input_path)

outs = []

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
    d_new = copy.deepcopy(d)

    obj_id = d_new["category_id"]
    obj_name = objects[obj_id - 1]
    category_id = obj2id[obj_name]

    d_new["category_id"] = category_id

    outs.append(d_new)

with PathManager.open(args.out_path, "w") as f:
    f.write(json.dumps(outs))
    f.flush()


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


# save_json(args.opath, outs)
