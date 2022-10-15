from multiprocessing.context import assert_spawning
import os

os.environ["PYOPENGL_PLATFORM"] = "egl"
import os.path as osp
import sys
import torch
import argparse
import time

import setproctitle
import cv2
import mmcv
import numpy as np
from tqdm import tqdm
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode

cur_dir = osp.abspath(osp.dirname(__file__))
PROJ_ROOT = osp.join(cur_dir, "../../../..")
sys.path.insert(0, PROJ_ROOT)
from lib.egl_renderer.egl_renderer_v3 import EGLRenderer
from lib.vis_utils.image import grid_show
from lib.pysixd import inout, misc
from lib.utils.mask_utils import cocosegm2mask
from core.gdrn_modeling.datasets.dataset_factory import register_datasets
from core.utils.data_utils import crop_resize_by_warp_affine, read_image_mmcv
from core.utils.camera_geometry import adapt_image_by_K
from lib.vis_utils.image import vis_image_mask_bbox_cv2
from core.utils.camera_geometry import get_K_crop_resize
import ref

im_H_ori = 400
im_W_ori = 400
target_H = 540
target_W = 720

new_K_list = [1075.65091572, 0.0, 360.0, 0.0, 1073.90347929, 270.0, 0.0, 0.0, 1.0]
new_K = np.array([1075.65091572, 0.0, 360.0, 0.0, 1073.90347929, 270.0, 0.0, 0.0, 1.0], dtype=np.float32).reshape(3, 3)

data_dir = osp.normpath(osp.join(PROJ_ROOT, "datasets/BOP_DATASETS/tless/train_primesense"))
out_data_dir = osp.normpath(osp.join(PROJ_ROOT, "datasets/BOP_DATASETS/tless/train_primesense_rescaled"))
ref_key = "tless"
data_ref = ref.__dict__[ref_key]
model_paths = data_ref.model_paths
objects = data_ref.objects
scene_ids = [f"{i:06d}" for i in range(25, 31)]


class ResizeTless(object):
    def __init__(self) -> None:
        pass

    def main(self):
        for scene_id in tqdm(scene_ids):
            scene_root = osp.join(data_dir, scene_id)
            new_scene_root = osp.join(out_data_dir, scene_id)

            new_rgb_dir = osp.join(new_scene_root, "rgb")
            new_mask_dir = osp.join(new_scene_root, "mask")
            new_visib_dir = osp.join(new_scene_root, "mask_visib")

            mmcv.mkdir_or_exist(new_rgb_dir)
            mmcv.mkdir_or_exist(new_mask_dir)
            mmcv.mkdir_or_exist(new_visib_dir)

            gt_dict_path = osp.join(scene_root, "scene_gt.json")
            gt_dict = mmcv.load(osp.join(scene_root, "scene_gt.json"))
            gt_info_dict = mmcv.load(osp.join(scene_root, "scene_gt_info.json"))
            cam_dict = mmcv.load(osp.join(scene_root, "scene_camera.json"))

            # copy scene_gt dict
            new_gt_path = osp.join(new_scene_root, "scene_gt.json")
            new_gt_info_dict_path = osp.join(new_scene_root, "scene_gt_info.json")
            new_cam_dict_path = osp.join(new_scene_root, "scene_camera.json")
            os.system(f"cp {gt_dict_path} {new_gt_path}")

            new_gt_info_dict = {}
            new_cam_dict = {}

            for str_im_id in tqdm(gt_dict, postfix=f"{scene_id}"):
                int_im_id = int(str_im_id)
                rgb_path = osp.join(scene_root, "rgb/{:06d}.png").format(int_im_id)
                mask_path = osp.join(scene_root, "mask/{:06d}_000000.png").format(int_im_id)
                mask_visib_path = osp.join(scene_root, "mask_visib/{:06d}_000000.png").format(int_im_id)
                depth_path = osp.join(scene_root, "depth/{:06d}.png".format(int_im_id))

                scene_im_id = f"{scene_id}/{str_im_id}"
                K = np.array(cam_dict[str_im_id]["cam_K"], dtype=np.float32).reshape(3, 3)

                # get K diff
                K_diff = new_K - K
                ox_diff = K_diff[0, 0]
                oy_diff = K_diff[1, 1]
                cx_diff = K_diff[0, 2]
                cy_diff = K_diff[1, 2]
                px_diff = int(np.round(cx_diff))
                py_diff = int(np.round(cy_diff))
                assert ox_diff < 0.001 and oy_diff < 0.001

                M = np.float32([[1, 0, px_diff], [0, 1, py_diff]])

                depth_scale = cam_dict[str_im_id]["depth_scale"]  # 10000

                new_cam_dict[str_im_id] = {"cam_K": new_K_list, "depth_scale": depth_scale}

                assert len(gt_dict[str_im_id]) == 1
                anno = gt_dict[str_im_id][0]

                bbox_visib = gt_info_dict[str_im_id][0]["bbox_visib"]
                bbox_obj = gt_info_dict[str_im_id][0]["bbox_obj"]

                # adjust bbox
                x, y, w, h = bbox_visib
                ax, ay, aw, ah = bbox_obj
                bbox_visib_new = [x + px_diff, y + py_diff, w, h]
                bbox_obj_new = [ax + px_diff, ay + py_diff, aw, ah]

                new_gt_info_dict[str_im_id] = gt_info_dict[str_im_id]
                new_gt_info_dict[str_im_id][0]["bbox_obj"] = bbox_obj_new
                new_gt_info_dict[str_im_id][0]["bbox_visib"] = bbox_visib_new

                # align rgb
                raw_img = read_image_mmcv(rgb_path, format="BGR")
                pad_img = np.zeros((target_H, target_W, 3))
                pad_img[:400, :400, :] = raw_img
                pad_img = cv2.warpAffine(pad_img, M, (target_W, target_H))
                new_rgb_path = osp.join(new_scene_root, "rgb/{:06d}.png").format(int_im_id)
                mmcv.imwrite(pad_img, new_rgb_path)

                # align mask
                raw_mask = mmcv.imread(mask_path, "unchanged")
                pad_mask = np.zeros((target_H, target_W, 1))
                pad_mask[:400, :400, 0] = raw_mask
                pad_mask = cv2.warpAffine(pad_mask, M, (target_W, target_H))
                new_mask_path = osp.join(new_scene_root, "mask/{:06d}_000000.png").format(int_im_id)
                mmcv.imwrite(pad_mask, new_mask_path)

                # align mask visib
                raw_mask_visib = mmcv.imread(mask_visib_path, "unchanged")
                pad_mask_visib = np.zeros((target_H, target_W, 1))
                pad_mask_visib[:400, :400, 0] = raw_mask_visib
                pad_mask_visib = cv2.warpAffine(pad_mask_visib, M, (target_W, target_H))
                new_mask_visib_path = osp.join(new_scene_root, "mask_visib/{:06d}_000000.png").format(int_im_id)
                mmcv.imwrite(pad_mask_visib, new_mask_visib_path)

                # TODO: align depth

            # save
            inout.save_json(new_gt_info_dict_path, new_gt_info_dict)
            inout.save_json(new_cam_dict_path, new_cam_dict)


if __name__ == "__main__":
    import argparse
    import time

    import setproctitle
    import torch

    parser = argparse.ArgumentParser(description="gen tudl pbr xyz")
    parser.add_argument("--gpu", type=str, default="0", help="gpu")
    parser.add_argument("--vis", default=False, action="store_true", help="vis")
    args = parser.parse_args()

    VIS = args.vis

    device = torch.device(int(args.gpu))
    dtype = torch.float32
    tensor_kwargs = {"device": device, "dtype": dtype}

    T_begin = time.perf_counter()
    setproctitle.setproctitle("gen_img_crop")
    func = ResizeTless()
    func.main()
    T_end = time.perf_counter() - T_begin
    print("total time: ", T_end)
