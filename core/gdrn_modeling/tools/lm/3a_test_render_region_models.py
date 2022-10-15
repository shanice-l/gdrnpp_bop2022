import random
import glob
import cv2
import time
from tqdm import tqdm
from transforms3d.axangles import axangle2mat
import matplotlib.pyplot as plt
import os.path as osp
import sys
import numpy as np

cur_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(cur_dir, "../../../../"))


from lib.vis_utils.image import vis_image_mask_bbox_cv2, grid_show
from lib.pysixd import inout, misc
from lib.render_vispy.model3d import load_models
from lib.render_vispy.renderer import Renderer


def proj_region(model_points_, colors, R, T, K, bg_label, height=480, width=640):
    # directly project 3d points onto 2d plane
    # numerical error due to round
    points_2d, z = misc.points_to_2D(model_points_, R, T, K)
    image_points = np.round(points_2d).astype(np.int32)
    ProjEmb = bg_label * np.ones((height, width)).astype(np.float32)
    depth = np.zeros((height, width, 1)).astype(np.float32)
    for i, (x, y) in enumerate(image_points):
        if x >= width or y >= height or x < 0 or y < 0:
            continue
        if depth[y, x, 0] == 0:
            depth[y, x, 0] = z[i]
            ProjEmb[y, x] = colors[i][0]
        elif z[i] < depth[y, x, 0]:
            depth[y, x, 0] = z[i]
            ProjEmb[y, x] = colors[i][0]
        else:
            pass
    return ProjEmb


def region_id_to_color(region_ids, num_fps):
    if num_fps <= 24:
        region_ids = (region_ids + 1) * 10
    elif num_fps > 24 and num_fps <= 50:
        region_ids = (region_ids + 1) * 5
    elif num_fps > 50 and num_fps <= 84:
        region_ids = (region_ids + 1) * 3
    elif num_fps > 84 and num_fps <= 126:
        region_ids = (region_ids + 1) * 2
    elif num_fps > 126 and num_fps <= 254:
        region_ids = region_ids + 1
    else:
        pass
        # region_ids = region_ids
    return region_ids


def color_to_region_id(seg_map, num_fps):
    if num_fps <= 24:
        # region_ids = (region_ids + 1) * 10
        seg_map = np.round(seg_map / 10)
    elif num_fps > 24 and num_fps <= 50:
        # region_ids = (region_ids + 1) * 5
        seg_map = np.round(seg_map / 5)
    elif num_fps > 50 and num_fps <= 84:
        # region_ids = (region_ids + 1) * 3
        seg_map = np.round(seg_map / 3)
    elif num_fps > 84 and num_fps <= 126:
        # region_ids = (region_ids + 1) * 2
        seg_map = np.round(seg_map / 2)
    elif num_fps > 126 and num_fps <= 254:
        pass
        # region_ids = region_ids
    seg_map[seg_map >= num_fps] = num_fps  # clip
    return seg_map


random.seed(0)

width = 640
height = 480
znear = 0.25
zfar = 6.0
K = np.array([[572.4114, 0, 325.2611], [0, 573.57043, 242.04899], [0, 0, 1]])
idx2class = {
    1: "ape",
    2: "benchvise",
    3: "bowl",
    4: "camera",
    5: "can",
    6: "cat",
    7: "cup",
    8: "driller",
    9: "duck",
    10: "eggbox",
    11: "glue",
    12: "holepuncher",
    13: "iron",
    14: "lamp",
    15: "phone",
}
classes = idx2class.values()
classes = sorted(classes)

num_fps = 8
model_root = "datasets/BOP_DATASETS/lm/models_fps{}/".format(num_fps)

model_paths = [osp.join(model_root, "obj_{:06d}.ply".format(cls_idx)) for cls_idx in idx2class]
models = [inout.load_ply(model_path, vertex_scale=1) for model_path in model_paths]
ren_models = load_models(
    model_paths,
    scale_to_meter=1,
    cache_dir=".cache",
    texture_paths=None,
    center=False,
    use_cache=True,
)

ren = Renderer(size=(width, height), cam=K)


# render target pose
R1 = axangle2mat((1, 0, 0), angle=0.5 * np.pi)
R2 = axangle2mat((0, 0, 1), angle=-0.7 * np.pi)
R = np.dot(R1, R2)
t = np.array([-0.1, 0.1, 0.7], dtype=np.float32)
pose = np.hstack([R, t.reshape((3, 1))])
# pose1 = np.hstack([R, 0.1 + t.reshape((3, 1))])
# pose2 = np.hstack([R, t.reshape((3, 1)) - 0.1])
# pose3 = np.hstack([R, t.reshape((3, 1)) - 0.05])
# pose4 = np.hstack([R, t.reshape((3, 1)) + 0.05])

# rendering
# NOTE: obj_id is 0-based
# BG = 255 * np.ones((height, width, 3), dtype=np.uint8)
for obj_id, cls_name in enumerate(classes):
    t0 = time.perf_counter()
    light_pos = np.random.uniform(-0.5, 0.5, 3)
    intensity = np.random.uniform(0.8, 2)
    light_color = intensity * np.random.uniform(0.9, 1.1, 3)
    # poses = [pose, pose1, pose2, pose3, pose4]
    # obj_ids = [obj_id, obj_id, obj_id, obj_id, obj_id]
    poses = [pose]
    obj_ids = [obj_id]

    ren.clear()
    # ren.draw_background(BG)
    for _i, _obj_id in enumerate(obj_ids):
        pose = poses[_i]
        ren.draw_model(ren_models[_obj_id], pose, rot_type="mat")
    ren_im, ren_depth = ren.finish(to_255=False)  # bgr
    ren_im = ren_im * 255
    seg_label = color_to_region_id(ren_im[:, :, 2], num_fps).astype("uint8")

    # seg_label[seg_label == 255] = num_fps
    # show_ims = [ren_im, ren_depth, seg_label]
    # show_titles = ['ren_im', 'depth', 'seg_label']
    # grid_show(show_ims, show_titles, row=1, col=3)
    print("seg_label: ", np.unique(seg_label))

    seg_proj = proj_region(models[_obj_id]["pts"], models[_obj_id]["colors"], R, t, K, bg_label=0)
    seg_proj = color_to_region_id(seg_proj, num_fps).astype("uint8")
    print("seg_proj: ", np.unique(seg_proj))

    grid_show(
        [seg_label, ren_depth, seg_proj],
        ["seg_label", "depth", "seg_proj"],
        row=1,
        col=3,
    )
