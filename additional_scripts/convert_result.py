import argparse
import os
import random
import re
import subprocess
import sys
from pathlib import Path

import torch
from detector import concatenate
from tqdm import tqdm
import numpy as np

PROJECT_DIR = Path('.').resolve()
LOCAL_DATA_DIR = PROJECT_DIR / Path("local_data")
TOOLKIT_DIR = PROJECT_DIR / Path('additional_scripts') / 'bop_toolkit_challenge'
assert TOOLKIT_DIR.exists()
EVAL_SCRIPT_PATH = TOOLKIT_DIR / 'scripts/eval_bop19.py'
DUMMY_EVAL_SCRIPT_PATH = TOOLKIT_DIR / 'scripts/eval_bop19_dummy.py'

sys.path.append(TOOLKIT_DIR.as_posix())
from bop_toolkit_lib import inout  # noqa


def join_tar_files(dir_path):
    assert dir_path.exists()
    files = list(dir_path.glob("*.tar"))
    print(f"there are {len(files)} .tar files in {dir_path}")
    assert len(files) > 0, dir_path
    files_with_data = []
    for file in tqdm(files, desc="Combining .tar files"):
        df = torch.load(file)['predictions']['maskrcnn_detections/refiner']
        start, end = map(int, re.compile(".*_([0-9]+)_([0-9]+)_.*").fullmatch(file.stem).groups())
        files_with_data.append((start, end, df, file))
    files_with_data = sorted(files_with_data)
    data = concatenate([f for _,_,f,_ in files_with_data])
    all_scenes = set(data.infos['scene_id'].tolist())
    all_views = set(zip(data.infos['scene_id'].tolist(), data.infos['view_id'].tolist()))
    all_label = set(data.infos['label'].tolist())
    print(f"There are {len(data)} entries. {len(all_scenes)} unique scenes. {len(all_views)} unique views. {len(all_label)} unique objects.")
    old_end = 0
    while len(files_with_data) > 0: # Checking that no files are missing
        start, end, _, file = files_with_data.pop(0)
        assert start == old_end, f"File is missing: {old_end}"
        old_end = end
    print("No files are missing")
    return data


def main():
    parser = argparse.ArgumentParser('Bop evaluation')
    parser.add_argument('--tar_dir', type=str, required=True)
    parser.add_argument('--dataset', default='', type=str, required=True)
    parser.add_argument('--result_name', default='', type=str)
    args = parser.parse_args()
    args.dummy = False
    args.convert_only = False
    csv_path = LOCAL_DATA_DIR / 'bop_predictions_csv'
    csv_path.mkdir(exist_ok=True)
    if len(args.result_name) == 0:
        result_id_int = random.randint(1e8, 1e9-1)
    else:
        result_id_int = args.result_name
    print(f"Using result id: {result_id_int}")
    ds_name = args.dataset
    csv_path = csv_path / f'challenge2020-{result_id_int}_{ds_name}-test.csv'
    csv_path.parent.mkdir(exist_ok=True)
    args.csv_path = csv_path
    predictions = join_tar_files(Path(args.tar_dir))
    run_evaluation(args, predictions)


def run_evaluation(args, predictions):
    csv_path = args.csv_path
    convert_results(predictions, csv_path)


def convert_results(predictions, out_csv_path):#, method):
    preds = []
    for n in tqdm(range(len(predictions))):
        TCO_n = predictions.poses[n]
        t = TCO_n[:3, -1] * 1e3  # m -> mm conversion
        R = TCO_n[:3, :3]
        row = predictions.infos.iloc[n]
        obj_id = int(row.label.split('_')[-1])
        score = row.score
        time = row.time
        pred = dict(scene_id=row.scene_id,
                    im_id=row.view_id,
                    obj_id=obj_id,
                    score=score,
                    t=t, R=R, time=time)
        preds.append(pred)
    # process time
    times = {}
    for item in preds:
        im_key = "{}/{}".format(item["scene_id"], item["im_id"])
        if im_key not in times:
            times[im_key] = []
        times[im_key].append(item["time"])

    for item in preds:
        im_key = "{}/{}".format(item["scene_id"], item["im_id"])
        curr_time = float(np.max(times[im_key]))
        item["time"] = curr_time

    print("Wrote:", out_csv_path)
    inout.save_bop_results(out_csv_path, preds)
    return out_csv_path


def run_bop_evaluation(filename, dummy=False):
    myenv = os.environ.copy()
    myenv['PYTHONPATH'] = TOOLKIT_DIR.as_posix()
    myenv['COSYPOSE_DIR'] = PROJECT_DIR.as_posix()
    if dummy:
        script_path = DUMMY_EVAL_SCRIPT_PATH
    else:
        script_path = EVAL_SCRIPT_PATH
    subprocess.call(['python', script_path.as_posix(),
                     '--renderer_type', 'cpp',
                     '--result_filenames', filename],
                    env=myenv, cwd=TOOLKIT_DIR.as_posix())


if __name__ == '__main__':
    main()
