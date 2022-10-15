import mmcv
import os.path as osp
import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
import random

cur_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(cur_dir, "../../../../"))

from lib.vis_utils.image import (
    vis_bbox_opencv,
    vis_image_bboxes_cv2,
    grid_show,
)
from lib.vis_utils.colormap import colormap
from core.utils.my_visualizer import MyVisualizer
from lib.utils.mask_utils import mask2bbox_xyxy
from core.utils.data_utils import crop_resize_by_warp_affine


DZI_SCALE_RATIO = 0.25
DZI_SHIFT_RATIO = 0.25
DZI_PAD_SCALE = 1.5

out_size = 256


colors = colormap(rgb=False, maximum=255)


im_path = osp.join(cur_dir, "../../../../datasets/BOP_DATASETS/lm/test/000009/rgb/000499.png")
assert osp.exists(im_path), im_path
print(im_path)
im = mmcv.imread(im_path, "color")


mask_visib_path = osp.join(
    cur_dir,
    "../../../../datasets/BOP_DATASETS/lm/test/000009/mask_visib/000499_000000.png",
)
mask_visib = mmcv.imread(mask_visib_path, "unchanged")

mask_full_path = osp.join(
    cur_dir,
    "../../../../datasets/BOP_DATASETS/lm/test/000009/mask/000499_000000.png",
)
mask_full = mmcv.imread(mask_full_path, "unchanged")

mask = mask_full

bbox_xyxy = mask2bbox_xyxy(mask)
bbox_xyxy[0] += 2
bbox_xyxy[2] += 2

x1, y1, x2, y2 = bbox_xyxy.copy()
cx = 0.5 * (x1 + x2)
cy = 0.5 * (y1 + y2)
bh = y2 - y1
bw = x2 - x1

# num = 8
# new_bboxes = []

# for i in range(num):
#     scale_ratio = 1 + DZI_SCALE_RATIO * (2 * np.random.random_sample() - 1)  # [1-0.25, 1+0.25]
#     shift_ratio = DZI_SHIFT_RATIO * (2 * np.random.random_sample(2) - 1)  # [-0.25, 0.25]
#     bbox_center = np.array([cx + bw * shift_ratio[0], cy + bh * shift_ratio[1]])  # (h/2, w/2)
#     scale = max(y2 - y1, x2 - x1) * scale_ratio * DZI_PAD_SCALE

#     new_x1 = bbox_center[0] - scale / 2
#     new_y1 = bbox_center[1] - scale / 2
#     new_x2 = bbox_center[0] + scale / 2
#     new_y2 = bbox_center[1] + scale / 2

#     new_bboxes.append(np.array([new_x1, new_y1, new_x2, new_y2]))

# new_bboxes = np.array(new_bboxes)

# maxx, maxy, minx, miny = 0, 0, 1000, 1000
# for i in range(len(new_bboxes)):
#     maxx, maxy, minx, miny = (
#         max(maxx, new_bboxes[i][0]),
#         max(maxy, new_bboxes[i][1]),
#         min(minx, new_bboxes[i][0]),
#         min(miny, new_bboxes[i][1]),
#     )
#     maxx, maxy, minx, miny = (
#         max(maxx, new_bboxes[i][2]),
#         max(maxy, new_bboxes[i][3]),
#         min(minx, new_bboxes[i][2]),
#         min(miny, new_bboxes[i][3]),
#     )

# center = np.array([(minx + maxx) / 2, (miny + maxy) / 2])
# scale = max(maxx - minx, maxy - miny) + 20

# new_bboxes[:, 0] = (new_bboxes[:, 0] - (center[0] - scale / 2)) * out_size / scale
# new_bboxes[:, 1] = (new_bboxes[:, 1] - (center[1] - scale / 2)) * out_size / scale
# new_bboxes[:, 2] = (new_bboxes[:, 2] - (center[0] - scale / 2)) * out_size / scale
# new_bboxes[:, 3] = (new_bboxes[:, 3] - (center[1] - scale / 2)) * out_size / scale

# bbox_xyxy[0] = (bbox_xyxy[0] - (center[0] - scale / 2)) * out_size / scale
# bbox_xyxy[1] = (bbox_xyxy[1] - (center[1] - scale / 2)) * out_size / scale
# bbox_xyxy[2] = (bbox_xyxy[2] - (center[0] - scale / 2)) * out_size / scale
# bbox_xyxy[3] = (bbox_xyxy[3] - (center[1] - scale / 2)) * out_size / scale


# im_zoom = crop_resize_by_warp_affine(im, center, scale, out_size)


im_vis = im.copy()

im_vis = vis_image_bboxes_cv2(im_vis, [bbox_xyxy], draw_center=False, box_thickness=4)
save_path = osp.expanduser(osp.join("~/vis_gdr_net/", "vis_DZI_im_ori.png"))
mmcv.mkdir_or_exist(osp.dirname(save_path))
print(save_path)
mmcv.imwrite(im_vis, save_path)

grid_show([im[:, :, ::-1], im_vis[:, :, ::-1]], ["im_ori", "im_vis"], row=1, col=2)
