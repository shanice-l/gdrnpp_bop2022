# -*- coding: utf-8 -*-
import argparse
import os
import os.path as osp
import numpy as np

# import pandas as pd
from tqdm import tqdm
import mmcv

# import sys
# cur_dir = osp.dirname(osp.abspath(__file__))
# sys.path.insert(0, osp.join(cur_dir, ".."))


def main():
    parser = argparse.ArgumentParser(description="Merge bop single obj results")
    parser.add_argument(
        "paths",
        nargs="+",
        help="paths to the single obj bop results csv files",
    )
    parser.add_argument("--res_path", help="path to the merged csv files", required=True)

    args = parser.parse_args()
    print("input files: ", args.paths)
    print("number of input files: ", len(args.paths))

    # merge files
    lines = []
    for i, _path in enumerate(tqdm(args.paths)):
        with open(_path, "r") as f:
            for j, line in enumerate(f):
                if j == 0:
                    if i == 0:
                        header = line.strip("\r\n")
                        lines.append(header)
                    else:
                        continue
                else:
                    lines.append(line.strip("\r\n"))

    mmcv.mkdir_or_exist(osp.dirname(args.res_path))

    print("Writing merged file...")
    with open(args.res_path, "w") as f:
        for line in lines:
            f.write("{}\n".format(line))
    print("Done. The merged results file has been saved to {}".format(args.res_path))


if __name__ == "__main__":
    main()
