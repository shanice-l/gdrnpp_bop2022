# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

# ref: https://github.com/thodan/sixd_toolkit/blob/master/params/dataset_params.py
# modified

import math
import os.path as osp
import sys

cur_dir = osp.dirname(osp.abspath(__file__))
from os.path import join as pjoin

sys.path.insert(0, osp.join(cur_dir, "../.."))

from . import inout


def get_dataset_params(name, model_type="", train_type="", test_type="", cam_type=""):

    p = {
        "name": name,
        "model_type": model_type,
        "train_type": train_type,
        "test_type": test_type,
        "cam_type": cam_type,
    }

    # Path to the folder with datasets
    common_base_path = osp.join(cur_dir, "../../data/SIXD_DATASETS/")

    # Path to the T-LESS Toolkit (https://github.com/thodan/t-less_toolkit)
    tless_tk_path = osp.join(cur_dir, "../../lib/t-less_toolkit/")

    if name in ["hinterstoisser", "linemod", "linemod_occ"]:  # LM6d_origin
        p["obj_count"] = 15
        p["scene_count"] = 15
        p["train_im_size"] = (640, 480)
        p["test_im_size"] = (640, 480)
        p["base_path"] = pjoin(common_base_path, "hinterstoisser")
        p["im_id_pad"] = 4
        p["model_texture_mpath"] = None
        p["cam_params_path"] = pjoin(p["base_path"], "camera.yml")

        # p['test_obj_depth_range'] = (600.90, 1102.35) # [mm] - with original GT
        p["test_obj_depth_range"] = (
            346.31,
            1499.84,
        )  # [mm] - with extended GT
        # (there are only 3 occurrences under 400 mm)

        p["test_obj_azimuth_range"] = (0, 2 * math.pi)
        p["test_obj_elev_range"] = (0, 0.5 * math.pi)

    elif name == "tless":
        p["obj_count"] = 30
        p["scene_count"] = 20

        p["base_path"] = pjoin(common_base_path, "t-less", "t-less_v2")
        p["im_id_pad"] = 4
        p["model_texture_mpath"] = None

        if p["model_type"] == "":
            p["model_type"] = "cad"
        if p["train_type"] == "":
            p["train_type"] = "primesense"
        if p["test_type"] == "":
            p["test_type"] = "primesense"
        if p["cam_type"] == "":
            p["cam_type"] = "primesense"

        p["cam_params_path"] = pjoin(tless_tk_path, "cam", "camera_" + p["cam_type"] + ".yml")
        if p["test_type"] in ["primesense", "kinect"]:
            p["test_im_size"] = (720, 540)
        elif p["test_type"] == "canon":
            p["test_im_size"] = (2560, 1920)

        if p["train_type"] in ["primesense", "kinect"]:
            p["train_im_size"] = (400, 400)
        elif p["train_type"] == "canon":
            p["train_im_size"] = (1900, 1900)
        elif p["train_type"] == "render_reconst":
            p["train_im_size"] = (1280, 1024)

        if p["test_type"] == "primesense":
            p["test_obj_depth_range"] = (649.89, 940.04)  # [mm]
            p["test_obj_azimuth_range"] = (0, 2 * math.pi)
            p["test_obj_elev_range"] = (-0.5 * math.pi, 0.5 * math.pi)
        if p["test_type"] == "kinect":
            # Not calculated yet
            p["test_depth_range"] = None
            p["test_obj_azimuth_range"] = None
            p["test_obj_elev_range"] = None
        elif p["test_type"] == "canon":
            # Not calculated yet
            p["test_depth_range"] = None
            p["test_obj_azimuth_range"] = None
            p["test_obj_elev_range"] = None

    elif name == "tudlight":
        p["obj_count"] = 3
        p["scene_count"] = 3
        p["train_im_size"] = (640, 480)
        p["test_im_size"] = (640, 480)
        p["base_path"] = pjoin(common_base_path, "tudlight")
        p["im_id_pad"] = 5  # 5
        p["model_texture_mpath"] = None
        p["cam_params_path"] = pjoin(p["base_path"], "camera.yml")

        p["test_obj_depth_range"] = (851.29, 2016.14)  # [mm]
        p["test_obj_azimuth_range"] = (0, 2 * math.pi)
        p["test_obj_elev_range"] = (-0.4363, 0.5 * math.pi)  # (-25, 90) [deg]

    elif name == "toyotalight":
        p["obj_count"] = 21
        p["scene_count"] = 21
        p["train_im_size"] = (640, 480)
        p["test_im_size"] = (640, 480)
        p["base_path"] = pjoin(common_base_path, "toyotalight")
        p["im_id_pad"] = 4  # 5
        p["model_texture_mpath"] = None
        p["cam_params_path"] = pjoin(p["base_path"], "camera.yml")

        # p['test_obj_depth_range'] = None # [mm]
        # p['test_obj_azimuth_range'] = None
        # p['test_obj_elev_range'] = None # (-25, 90) [deg]

    elif name == "rutgers":
        p["obj_count"] = 14
        p["scene_count"] = 14
        p["train_im_size"] = (640, 480)
        p["test_im_size"] = (640, 480)
        p["base_path"] = pjoin(common_base_path, "rutgers")
        p["im_id_pad"] = 4
        p["model_texture_mpath"] = pjoin(p["base_path"], "models", "obj_{:02d}.png")
        p["cam_params_path"] = pjoin(p["base_path"], "camera.yml")

        p["test_obj_depth_range"] = (594.41, 739.12)  # [mm]
        p["test_obj_azimuth_range"] = (0, 2 * math.pi)
        p["test_obj_elev_range"] = (-0.5 * math.pi, 0.5 * math.pi)

    elif name == "tejani":
        p["obj_count"] = 6
        p["scene_count"] = 6
        p["train_im_size"] = (640, 480)
        p["test_im_size"] = (640, 480)
        p["base_path"] = pjoin(common_base_path, "tejani")
        p["im_id_pad"] = 4
        p["model_texture_mpath"] = None
        p["cam_params_path"] = pjoin(p["base_path"], "camera.yml")

        p["test_obj_depth_range"] = 509.12 - 1120.41  # [mm]
        p["test_obj_azimuth_range"] = (0, 2 * math.pi)
        p["test_obj_elev_range"] = (0, 0.5 * math.pi)

    elif name == "doumanoglou":
        p["obj_count"] = 2
        p["scene_count"] = 3
        p["train_im_size"] = (640, 480)
        p["test_im_size"] = (640, 480)
        p["base_path"] = pjoin(common_base_path, "doumanoglou")
        p["im_id_pad"] = 4
        p["model_texture_mpath"] = None
        p["cam_params_path"] = pjoin(p["base_path"], "camera.yml")

        p["test_obj_depth_range"] = 454.56 - 1076.29  # [mm]
        p["test_obj_azimuth_range"] = (0, 2 * math.pi)
        p["test_obj_elev_range"] = (-1.0297, 0.5 * math.pi)  # (-59, 90) [deg]

    else:
        print("Error: Unknown SIXD dataset {}".format(name))
        exit(-1)

    models_dir = "models" if p["model_type"] == "" else "models_" + p["model_type"]
    train_dir = "train" if p["train_type"] == "" else "train_" + p["train_type"]
    test_dir = "test" if p["test_type"] == "" else "test_" + p["test_type"]

    # Image ID format
    im_id_f = "{:" + str(p["im_id_pad"]).zfill(2) + "d}"

    # Paths and path masks
    p["model_mpath"] = pjoin(p["base_path"], models_dir, "obj_{:02d}.ply")
    p["models_info_path"] = pjoin(p["base_path"], models_dir, "models_info.yml")

    p["obj_info_mpath"] = pjoin(p["base_path"], train_dir, "{:02d}", "info.yml")
    p["obj_gt_mpath"] = pjoin(p["base_path"], train_dir, "{:02d}", "gt.yml")
    p["obj_gt_stats_mpath"] = pjoin(p["base_path"], train_dir + "_gt_stats", "{:02d}_delta={}.yml")
    p["train_rgb_mpath"] = pjoin(p["base_path"], train_dir, "{:02d}", "rgb", im_id_f + ".png")
    p["train_depth_mpath"] = pjoin(p["base_path"], train_dir, "{:02d}", "depth", im_id_f + ".png")
    p["train_mask_mpath"] = pjoin(p["base_path"], train_dir, "{:02d}", "mask", im_id_f + "_{:02d}.png")
    p["train_mask_visib_mpath"] = pjoin(
        p["base_path"],
        train_dir,
        "{:02d}",
        "mask_visib",
        im_id_f + "_{:02d}.png",
    )

    p["scene_info_mpath"] = pjoin(p["base_path"], test_dir, "{:02d}", "info.yml")
    p["scene_gt_mpath"] = pjoin(p["base_path"], test_dir, "{:02d}", "gt.yml")
    p["scene_gt_stats_mpath"] = pjoin(p["base_path"], test_dir + "_gt_stats", "{:02d}_delta={}.yml")
    p["test_rgb_mpath"] = pjoin(p["base_path"], test_dir, "{:02d}", "rgb", im_id_f + ".png")
    p["test_depth_mpath"] = pjoin(p["base_path"], test_dir, "{:02d}", "depth", im_id_f + ".png")
    p["test_mask_mpath"] = pjoin(p["base_path"], test_dir, "{:02d}", "mask", im_id_f + "_{:02d}.png")
    p["test_mask_visib_mpath"] = pjoin(
        p["base_path"],
        test_dir,
        "{:02d}",
        "mask_visib",
        im_id_f + "_{:02d}.png",
    )

    p["test_set_fpath"] = pjoin(p["base_path"], "test_set_v1.yml")

    # NOTE: split by bb8
    p["test_set_bb8_fpath"] = pjoin(p["base_path"], "test_set_bb8_sixd.yml")

    p["cam"] = inout.load_cam_params(p["cam_params_path"])

    return p
