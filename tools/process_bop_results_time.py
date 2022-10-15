# -*- coding: utf-8 -*-
import argparse
import os
import os.path as osp
import numpy as np
import pandas as pd

# import sys
# cur_dir = osp.dirname(osp.abspath(__file__))
# sys.path.insert(0, osp.join(cur_dir, ".."))


def load_predicted_csv(fname):
    df = pd.read_csv(fname)
    info_list = df.to_dict("records")
    return info_list


def parse_Rt_in_csv(item):
    return np.array([float(i) for i in item.split(" ")])


def write_result(fname, results):
    with open(fname, "w") as f:
        f.write("scene_id,im_id,obj_id,score,R,t,time\n")
        for item in results:
            f.write("{:d},".format(item["scene_id"]))
            f.write("{:d},".format(item["im_id"]))
            f.write("{:d},".format(item["obj_id"]))
            f.write("{:f},".format(item["score"]))

            if isinstance(item["R"], np.ndarray):
                R_list = item["R"].flatten().tolist()
            else:
                R_list = item["R"]
            for i, r in enumerate(R_list):
                sup = " " if i != 8 else ", "
                f.write("{:f}{}".format(r, sup))

            if isinstance(item["t"], np.ndarray):
                t_list = item["t"].flatten().tolist()
            else:
                t_list = item["t"]
            for i, t_item in enumerate(t_list):
                sup = " " if i != 2 else ", "
                f.write("{:f}{}".format(t_item, sup))

            f.write("{:f}\n".format(item["time"]))


def main():
    parser = argparse.ArgumentParser(description="Process time of the bop results file")
    parser.add_argument("path", help="path to the bop results csv file")

    args = parser.parse_args()
    print("input file: ", args.path)
    assert osp.exists(args.path), args.path
    assert args.path.endswith(".csv"), args.path

    results = load_predicted_csv(args.path)
    # backup old file
    os.system(f"cp -v {args.path} {args.path.replace('.csv', '.bak.csv')}")

    # process time
    times = {}
    for item in results:
        im_key = "{}/{}".format(item["scene_id"], item["im_id"])
        if im_key not in times:
            times[im_key] = []
        times[im_key].append(item["time"])

    for item in results:
        im_key = "{}/{}".format(item["scene_id"], item["im_id"])
        item["time"] = float(np.max(times[im_key]))
        item["R"] = parse_Rt_in_csv(item["R"]).reshape(3, 3)
        item["t"] = parse_Rt_in_csv(item["t"])

    write_result(args.path, results)
    print("Done. The results file has been saved to {}".format(args.path))


if __name__ == "__main__":
    main()
