import os.path as osp
import sys
import numpy as np
import torch

cur_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(cur_dir, "../../../../"))
import mmcv
import ref
from core.utils.utils import get_emb_show
from lib.vis_utils.image import grid_show
from core.utils.data_utils import xyz_to_region


model_dir = ref.lm_full.model_dir
id2obj = ref.lm_full.id2obj
im_H = 480
im_W = 640

fps_dict = mmcv.load(osp.join(model_dir, "fps_points.pkl"))

num_fps = 32
obj_id = 1
im_id = 10
####################################################


xyz_path = osp.join(
    ref.lm_full.dataset_root,
    f"test/xyz_crop/{obj_id:06d}/{im_id:06d}_000000.pkl",
)
xyz_info = mmcv.load(xyz_path)
x1, y1, x2, y2 = xyz_info["xyxy"]
xyz_crop = xyz_info["xyz_crop"]
xyz = np.zeros((im_H, im_W, 3), dtype=np.float32)
xyz[y1 : y2 + 1, x1 : x2 + 1, :] = xyz_crop  # HW3


seg_label = np.zeros((im_H, im_W), dtype=np.uint8)
fps_points = fps_dict[str(obj_id)][f"fps{num_fps}_and_center"][:-1]  # fx3
seg_crop = xyz_to_region(xyz_crop, fps_points)
seg_label[y1 : y2 + 1, x1 : x2 + 1] = seg_crop
seg_label = seg_label.astype("uint8")

seg_label_th = torch.as_tensor(seg_label).to(torch.long)
onehot = torch.nn.functional.one_hot(seg_label_th, num_fps + 1)


grid_show(
    [seg_label, seg_crop, get_emb_show(xyz_crop), get_emb_show(xyz)],
    ["seg_label", "seg_crop", "xyz_crop", "xyz"],
    row=2,
    col=2,
)

# for i in range(1, num_fps + 1):
#     onehot_i = onehot[:, :, i].numpy()[y1 : y2 + 1, x1 : x2 + 1]
#     grid_show([seg_crop, (seg_crop == i).astype("uint8"), onehot_i], ["seg_crop", f"{i}", f"onehot_{i}"], row=1, col=3)
