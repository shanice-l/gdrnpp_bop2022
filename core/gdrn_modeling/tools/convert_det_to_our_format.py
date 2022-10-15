import mmcv
import sys
import argparse
import json
import os

parser = argparse.ArgumentParser(description="convert det from bop format to ours")
parser.add_argument("--idir", type=str, default="/data2/lxy/Storage/bop22_results/yolovx_amodal", help="input path")
parser.add_argument("--odir", type=str, default="datasets/BOP_DATASETS/", help="output path")
args = parser.parse_args()


def convert_format(ipath, opath):
    ds = mmcv.load(ipath)
    outs = {}
    for d in ds:
        scene_id = d["scene_id"]
        image_id = d["image_id"]
        scene_im_id = f"{scene_id}/{image_id}"

        obj_id = d["category_id"]
        score = d["score"]

        bbox = d["bbox"]
        time = d["time"]

        cur_dict = {
            "bbox_est": bbox,
            "obj_id": obj_id,
            "score": score,
            "time": time,
        }

        if scene_im_id in outs.keys():
            outs[scene_im_id].append(cur_dict)
        else:
            outs[scene_im_id] = [cur_dict]

    save_json(opath, outs)
    print(f"json file has been saved at {opath}")


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


if __name__ == "__main__":
    dsets = sorted(os.listdir(args.idir))
    for dset in dsets:
        json_files = os.listdir(os.path.join(args.idir, dset))
        for json_file in json_files:
            ipath = os.path.join(args.idir, dset, json_file)

            odir = os.path.join(args.odir, dset, "test/test_bboxes")
            mmcv.mkdir_or_exist(odir)
            opath = os.path.join(odir, json_file)

            convert_format(ipath, opath)
