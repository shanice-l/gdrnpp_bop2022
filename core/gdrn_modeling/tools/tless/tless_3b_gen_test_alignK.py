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
from lib.pysixd import misc
from lib.pysixd import inout, misc, transform

from lib.utils.mask_utils import cocosegm2mask
from core.gdrn_modeling.datasets.dataset_factory import register_datasets
from core.utils.data_utils import crop_resize_by_warp_affine, read_image_mmcv
from core.utils.camera_geometry import adapt_image_by_K
from lib.vis_utils.image import vis_image_bboxes_cv2
from core.utils.camera_geometry import get_K_crop_resize
import ref

target_H = 540
target_W = 720
near = 0.01
far = 6.5
new_K_list = [1075.65091572, 0.0, 360.0, 0.0, 1073.90347929, 270.0, 0.0, 0.0, 1.0]
new_K = np.array([1075.65091572, 0.0, 360.0, 0.0, 1073.90347929, 270.0, 0.0, 0.0, 1.0], dtype=np.float32).reshape(3, 3)

data_dir = osp.normpath(osp.join(PROJ_ROOT, "datasets/BOP_DATASETS/tless/test_primesense"))
out_data_dir = osp.normpath(osp.join(PROJ_ROOT, "datasets/BOP_DATASETS/tless/test_primesense_alignK"))
ref_key = "tless"
data_ref = ref.__dict__[ref_key]
model_paths = data_ref.model_paths
objects = data_ref.objects
scene_ids = [f"{i:06d}" for i in range(1, 21)]


class ResizeTless(object):
    def __init__(self) -> None:
        self.models = []
        for model_path in model_paths:
            model = inout.load_ply(model_path, vertex_scale=0.001)
            self.models.append(model)

    def main(self):
        for scene_id in tqdm(scene_ids):
            scene_root = osp.join(data_dir, scene_id)
            new_scene_root = osp.join(out_data_dir, scene_id)

            new_rgb_dir = osp.join(new_scene_root, "rgb")
            new_depth_dir = osp.join(new_scene_root, "depth")
            new_mask_dir = osp.join(new_scene_root, "mask")
            new_visib_dir = osp.join(new_scene_root, "mask_visib")

            mmcv.mkdir_or_exist(new_rgb_dir)
            mmcv.mkdir_or_exist(new_depth_dir)
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
                new_rgb_path = osp.join(new_scene_root, "rgb/{:06d}.png").format(int_im_id)

                depth_path = osp.join(scene_root, "depth/{:06d}.png".format(int_im_id))
                new_depth_path = osp.join(new_scene_root, "depth/{:06d}.png".format(int_im_id))

                scene_im_id = f"{scene_id}/{str_im_id}"
                K = np.array(cam_dict[str_im_id]["cam_K"], dtype=np.float32).reshape(3, 3)

                depth_scale = cam_dict[str_im_id]["depth_scale"]  # 0.1

                new_cam_dict[str_im_id] = {"cam_K": new_K_list, "depth_scale": depth_scale}

                # adjust
                raw_img = read_image_mmcv(rgb_path, format="BGR")
                adapt_img = adapt_image_by_K(raw_img, K_old=K, K_new=new_K, height=target_H, width=target_W)
                mmcv.imwrite(adapt_img, new_rgb_path)

                raw_depth = mmcv.imread(depth_path, "unchanged")
                adapt_depth = adapt_image_by_K(raw_depth, K_old=K, K_new=new_K, height=target_H, width=target_W)
                mmcv.imwrite(adapt_depth, new_depth_path)

                for i, anno in enumerate(gt_dict[str_im_id]):
                    obj_id = anno["obj_id"]

                    R = np.array(anno["cam_R_m2c"], dtype="float32").reshape(3, 3)
                    t = np.array(anno["cam_t_m2c"], dtype="float32") / 1000.0
                    pose = np.hstack([R, t.reshape(3, 1)])

                    mask_path = osp.join(scene_root, "mask/{:06d}_{:06d}.png").format(int_im_id, i)
                    mask_visib_path = osp.join(scene_root, "mask_visib/{:06d}_{:06d}.png").format(int_im_id, i)

                    new_mask_path = osp.join(new_scene_root, "mask/{:06d}_{:06d}.png").format(int_im_id, i)
                    new_mask_visib_path = osp.join(new_scene_root, "mask_visib/{:06d}_{:06d}.png").format(int_im_id, i)

                    raw_mask = mmcv.imread(mask_path, "unchanged")
                    adapt_mask = adapt_image_by_K(raw_mask, K_old=K, K_new=new_K, height=target_H, width=target_W)
                    mmcv.imwrite(adapt_mask, new_mask_path)

                    raw_mask_visib = mmcv.imread(mask_visib_path, "unchanged")
                    adapt_mask_visib = adapt_image_by_K(
                        raw_mask_visib, K_old=K, K_new=new_K, height=target_H, width=target_W
                    )
                    mmcv.imwrite(adapt_mask_visib, new_mask_visib_path)

                    # NOTE: gen bbox from pose, not reliable !!!
                    bbox_visib = bbox_obj = misc.compute_2d_bbox_xywh_from_pose(
                        self.models[obj_id - 1]["pts"], pose, K, width=target_W, height=target_H, clip=True
                    ).tolist()
                    if str_im_id not in new_gt_info_dict:
                        new_gt_info_dict[str_im_id] = []
                    new_gt_info_dict[str_im_id].append(gt_info_dict[str_im_id][i])
                    new_gt_info_dict[str_im_id][i]["bbox_obj"] = bbox_obj
                    new_gt_info_dict[str_im_id][i]["bbox_visib"] = bbox_visib

            # save
            inout.save_json(new_gt_info_dict_path, new_gt_info_dict)
            inout.save_json(new_cam_dict_path, new_cam_dict)


if __name__ == "__main__":
    import argparse
    import time

    import setproctitle
    import torch

    parser = argparse.ArgumentParser(description="None")
    parser.add_argument("--gpu", type=str, default="0", help="gpu")
    parser.add_argument("--vis", default=False, action="store_true", help="vis")
    args = parser.parse_args()

    VIS = args.vis

    device = torch.device(int(args.gpu))
    dtype = torch.float32
    tensor_kwargs = {"device": device, "dtype": dtype}

    T_begin = time.perf_counter()
    setproctitle.setproctitle("test_img_crop")
    func = ResizeTless()
    func.main()
    T_end = time.perf_counter() - T_begin
    print("total time: ", T_end)
