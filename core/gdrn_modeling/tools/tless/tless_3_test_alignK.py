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

target_H = 540
target_W = 720
near = 0.01
far = 6.5
new_K = np.array([1075.65091572, 0.0, 360.0, 0.0, 1073.90347929, 270.0, 0.0, 0.0, 1.0]).reshape(3, 3)

data_dir = osp.normpath(osp.join(PROJ_ROOT, "datasets/BOP_DATASETS/tless/test_primesense"))
texture_paths = None
ref_key = "tless"
data_ref = ref.__dict__[ref_key]
model_paths = data_ref.model_paths
objects = data_ref.objects

dataset_name = "tless_bop_test_primesense"
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
            for anno in annos:
                raw_mask = cocosegm2mask(anno["segmentation"], target_H, target_W)

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
                # adjust
                adapt_img = adapt_image_by_K(raw_img, K_old=K, K_new=new_K, height=target_H, width=target_W)
                adapt_mask = adapt_image_by_K(raw_mask, K_old=K, K_new=new_K, height=target_H, width=target_W)
                adapt_depth = adapt_image_by_K(raw_depth, K_old=K, K_new=new_K, height=target_H, width=target_W)
                color_diff = adapt_img - bgr_gl
                depth_diff = adapt_depth - depth_gl
                mask_diff = adapt_mask - mask_gl
                if args.vis:
                    grid_show(
                        [
                            color_diff,
                            bgr_gl,
                            adapt_img,
                            depth_diff,
                            depth_gl,
                            adapt_depth,
                            mask_diff,
                            mask_gl,
                            adapt_mask,
                        ],
                        [
                            "color_diff",
                            "render",
                            "adapt_real",
                            "depth_diff",
                            "ren_depth",
                            "adapt_depth",
                            "mask_diff",
                            "ren_mask",
                            "adapt_mask",
                        ],
                        row=3,
                        col=3,
                    )
                    # grid_show(
                    #     [color_diff, bgr_gl, adapt_img, mask_gl, adapt_mask, depth_gl, adapt_depth, ],
                    #     ["color_diff", "render", "adapt_img", "mask_gl", "adapt_mask", "depth_gl", "adapt_depth"],
                    #     row = 3,
                    #     col = 3
                    # )


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
