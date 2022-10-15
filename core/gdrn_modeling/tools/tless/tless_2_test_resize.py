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
from lib.utils.mask_utils import cocosegm2mask
from core.gdrn_modeling.datasets.dataset_factory import register_datasets
from core.utils.data_utils import crop_resize_by_warp_affine, read_image_mmcv
from core.utils.camera_geometry import adapt_image_by_K
from lib.vis_utils.image import vis_image_bboxes_cv2
from core.utils.camera_geometry import get_K_crop_resize
import ref

im_H_ori = 400
im_W_ori = 400
target_H = 540
target_W = 720

near = 0.01
far = 6.5
new_K = np.array([1075.65091572, 0.0, 360.0, 0.0, 1073.90347929, 270.0, 0.0, 0.0, 1.0]).reshape(3, 3)

data_dir = osp.normpath(osp.join(PROJ_ROOT, "datasets/BOP_DATASETS/tless/train_primesense"))
texture_paths = None
ref_key = "tless"
data_ref = ref.__dict__[ref_key]
model_paths = data_ref.model_paths
objects = data_ref.objects

dataset_name = "tless_train_primesense"
print(dataset_name)
register_datasets([dataset_name])

meta = MetadataCatalog.get(dataset_name)
print("MetadataCatalog: ", meta)
objs = meta.objs


class ResizeTless(object):
    def __init__(self) -> None:
        self.renderer = None

    def get_renderer(self):
        if self.renderer is None:
            self.renderer = EGLRenderer(
                model_paths,
                texture_paths=texture_paths,
                vertex_scale=0.001,
                height=target_H,
                width=target_W,
                znear=near,
                zfar=far,
                use_cache=True,
                gpu_id=int(args.gpu),
            )
            self.image_tensor = torch.cuda.FloatTensor(target_H, target_W, 4, device=device).detach()
            self.seg_tensor = torch.cuda.FloatTensor(target_H, target_W, 4, device=device).detach()
            self.pc_obj_tensor = torch.cuda.FloatTensor(target_H, target_W, 4, device=device).detach()
            self.pc_cam_tensor = torch.cuda.FloatTensor(target_H, target_W, 4, device=device).detach()
        return self.renderer

    def main(self):
        dset_dicts = DatasetCatalog.get(dataset_name)

        for d in tqdm(dset_dicts):
            K = d["cam"]
            file_name = d["file_name"]
            raw_img = read_image_mmcv(file_name, format="BGR")
            depth_path = d["depth_file"]
            raw_depth = mmcv.imread(depth_path, "unchanged") / d["depth_factor"]  # (400, 400)

            annos = d["annotations"]
            assert len(annos) == 1
            anno = annos[0]
            raw_mask = cocosegm2mask(anno["segmentation"], im_H_ori, im_W_ori)

            pose = anno["pose"]
            bbox_visib = anno["bbox"]  # xywh
            bbox_obj = anno["bbox_obj"]

            render_obj_id = anno["category_id"]

            self.get_renderer().render(
                [render_obj_id],
                [pose],
                K=new_K,
                image_tensor=self.image_tensor,
                seg_tensor=self.seg_tensor,
                # pc_obj_tensor=self.pc_obj_tensor,
                pc_cam_tensor=self.pc_cam_tensor,
            )

            bgr_gl = (self.image_tensor[:, :, :3].cpu().numpy() + 0.5).astype(np.uint8)
            mask_gl = (self.seg_tensor[:, :, 0] > 0).cpu().numpy().astype(np.uint8)
            depth_gl = self.pc_cam_tensor[:, :, 2].cpu().numpy()

            pad_img = np.zeros((target_H, target_W, 3))
            pad_mask = np.zeros((target_H, target_W, 1))
            pad_depth = np.zeros((target_H, target_W, 1))

            # adjust
            K_diff = new_K - K
            ox_diff = K_diff[0, 0]
            oy_diff = K_diff[1, 1]
            cx_diff = K_diff[0, 2]
            cy_diff = K_diff[1, 2]
            px_diff = int(np.round(cx_diff))
            py_diff = int(np.round(cy_diff))
            assert ox_diff < 0.001 and oy_diff < 0.001

            # adjust bbox
            x, y, w, h = bbox_visib
            ax, ay, aw, ah = bbox_obj
            bbox_visib_new = [x + px_diff, y + py_diff, w, h]
            bbox_visib_new_xyxy = [x + px_diff, y + py_diff, x + px_diff + w, y + py_diff + h]
            bbox_obj_new = [ax + px_diff, ay + py_diff, aw, ah]
            bbox_obj_new_xyxy = [ax + px_diff, ay + py_diff, ax + px_diff + aw, ay + py_diff + ah]

            pad_img[:400, :400, :] = raw_img
            pad_img = pad_img.astype(np.uint8)

            pad_mask[:400, :400, 0] = raw_mask
            pad_mask = pad_mask.astype(np.uint8)

            pad_depth[:400, :400, 0] = raw_depth * raw_mask

            # translate image
            M = np.float32([[1, 0, px_diff], [0, 1, py_diff]])

            pad_img = cv2.warpAffine(pad_img, M, (target_W, target_H))
            pad_mask = cv2.warpAffine(pad_mask, M, (target_W, target_H))
            pad_depth = cv2.warpAffine(pad_depth, M, (target_W, target_H))

            pad_img_vis = vis_image_bboxes_cv2(
                pad_img,
                [bbox_visib_new_xyxy],
                box_thickness=5,
            )
            bgr_gl_vis = vis_image_bboxes_cv2(
                bgr_gl,
                [bbox_obj_new_xyxy],
                box_thickness=5,
            )

            color_diff = bgr_gl + pad_img
            depth_diff = pad_depth - depth_gl
            mask_diff = pad_mask - mask_gl

            # adapt_img = adapt_image_by_K(
            #     raw_img,  K_old=K, K_new=new_K, height=target_H, width=target_W
            # )
            if args.vis:
                grid_show(
                    [
                        color_diff,
                        bgr_gl_vis,
                        pad_img_vis,
                        depth_diff,
                        depth_gl,
                        pad_depth,
                        mask_diff,
                        mask_gl,
                        pad_mask,
                    ],
                    [
                        "color_diff",
                        "render",
                        "pad_real",
                        "depth_diff",
                        "ren_depth",
                        "pad_depth",
                        "mask_diff",
                        "ren_mask",
                        "pad_mask",
                    ],
                    row=3,
                    col=3,
                )


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
    setproctitle.setproctitle("test_img_crop")
    func = ResizeTless()
    func.main()
    T_end = time.perf_counter() - T_begin
    print("total time: ", T_end)
