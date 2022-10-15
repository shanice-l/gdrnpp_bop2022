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
from lib.vis_utils.image import vis_image_mask_bbox_cv2
from core.utils.camera_geometry import get_K_crop_resize

idx2class = {1: "dragon", 2: "frog", 3: "can"}

class2idx = {_name: _id for _id, _name in idx2class.items()}

classes = idx2class.values()
classes = sorted(classes)

# DEPTH_FACTOR = 1000.
height = 480
width = 640
near = 0.01
far = 6.5

input_res = 256
out_res = 64

data_dir = osp.normpath(osp.join(PROJ_ROOT, "datasets/BOP_DATASETS/tudl/train_pbr"))

cls_indexes = [_idx for _idx in sorted(idx2class.keys())]
cls_names = [idx2class[cls_idx] for cls_idx in cls_indexes]
model_dir = osp.normpath(osp.join(PROJ_ROOT, "datasets/BOP_DATASETS/tudl/models"))
model_paths = [osp.join(model_dir, f"obj_{cls_idx:06d}.ply") for cls_idx in cls_indexes]
texture_paths = None

dataset_name = "tudl_train_pbr"
print(dataset_name)
register_datasets([dataset_name])

meta = MetadataCatalog.get(dataset_name)
print("MetadataCatalog: ", meta)
objs = meta.objs


def normalize_to_01(img):
    if img.max() != img.min():
        return (img - img.min()) / (img.max() - img.min())
    else:
        return img


def get_emb_show(bbox_emb):
    show_emb = bbox_emb.copy()
    show_emb = normalize_to_01(bbox_emb)
    return show_emb


class TestCrop(object):
    def __init__(self) -> None:
        self.renderer = None

    def get_renderer(self):
        if self.renderer is None:
            self.renderer = EGLRenderer(
                model_paths,
                texture_paths=texture_paths,
                vertex_scale=0.001,
                height=height,
                width=width,
                znear=near,
                zfar=far,
                use_cache=True,
                gpu_id=int(args.gpu),
            )
            self.image_tensor = torch.cuda.FloatTensor(height, width, 4, device=device).detach()
            self.seg_tensor = torch.cuda.FloatTensor(height, width, 4, device=device).detach()
            self.pc_obj_tensor = torch.cuda.FloatTensor(height, width, 4, device=device).detach()
            self.pc_cam_tensor = torch.cuda.FloatTensor(height, width, 4, device=device).detach()
        return self.renderer

    def main(self):
        dset_dicts = DatasetCatalog.get(dataset_name)

        for d in tqdm(dset_dicts):
            K = d["cam"]
            file_name = d["file_name"]
            img = read_image_mmcv(file_name, format="BGR")

            annos = d["annotations"]
            for anno in annos:
                mask = cocosegm2mask(anno["segmentation"], height, width)
                pose = anno["pose"]
                bbox_visib = anno["bbox"]
                bbox_mode = anno["bbox_mode"]
                bbox_obj = anno["bbox_obj"]
                label = objs[anno["category_id"]]
                x, y, w, h = bbox_obj

                # skip intruncted images
                if (x > 0 and x + w < width) or (y > 0 and y + h < height):
                    continue

                bbox_xyxy = BoxMode.convert(bbox_obj, bbox_mode, BoxMode.XYXY_ABS)
                x1, y1, x2, y2 = bbox_xyxy.copy()

                cx = 0.5 * (x1 + x2)
                cy = 0.5 * (y1 + y2)
                bh = y2 - y1
                bw = x2 - x1

                bbox_center = np.array([cx, cy])  # (w/2, h/2)
                scale = max(y2 - y1, x2 - x1)

                img_vis = vis_image_mask_bbox_cv2(
                    img,
                    [mask],
                    bboxes=[np.array(bbox_xyxy)],
                    labels=[label],
                )
                roi_img = crop_resize_by_warp_affine(img, bbox_center, scale, input_res, interpolation=cv2.INTER_LINEAR)

                # show roi img
                grid_show(
                    [roi_img, img_vis],
                    [
                        "roi_img",
                        "img",
                    ],
                    row=1,
                    col=2,
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
    test_crop = TestCrop()
    test_crop.main()
    T_end = time.perf_counter() - T_begin
    print("total time: ", T_end)
