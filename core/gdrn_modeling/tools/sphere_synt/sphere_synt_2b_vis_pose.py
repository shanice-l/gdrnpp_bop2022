import os.path as osp
import sys
from tqdm import tqdm
import numpy as np
import random
import mmcv

cur_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(cur_dir, "../../../../"))

from lib.pysixd import inout, misc, transform
import ref

# from lib.meshrenderer.meshrenderer_texture_color import Renderer
from lib.render_vispy.renderer import Renderer
from lib.render_vispy.model3d import load_models
from lib.vis_utils.image import grid_show


if len(sys.argv) > 1:
    split = sys.argv[1]
else:
    raise RuntimeError("Usage: python this_file.py <split>(train/test)")

print("split: ", split)


ref_key = "sphere_synt"
data_ref = ref.__dict__[ref_key]

model_dir = data_ref.model_dir
id2obj = data_ref.id2obj
K = data_ref.camera_matrix
height = 480
width = 640


# parameters
scene = 1
# N_grid = 200
if split == "train":

    scene_dir = osp.join(data_ref.train_dir, f"{scene:06d}")

    N_sample = 20000
    # minNoiseSigma = 0
    # maxNoiseSigma = 15
    # minOutlier = 0
    # maxOutlier = 0.3
else:

    scene_dir = osp.join(data_ref.test_dir, f"{scene:06d}")
    N_sample = 2000


trans_min = [-2, -2, 4]
trans_max = [2, 2, 8]


def main():

    vertex_scale = data_ref.vertex_scale
    obj_id = 1
    model_path = osp.join(model_dir, f"obj_{obj_id:06d}.ply")

    ren = Renderer(size=(width, height), cam=K)
    ren_models = load_models([model_path], scale_to_meter=vertex_scale)

    scene_gt_file = osp.join(scene_dir, "scene_gt.json")
    scene_gt_info_file = osp.join(scene_dir, "scene_gt_info.json")
    scene_gt_dict = mmcv.load(scene_gt_file)
    scene_gt_info_dict = mmcv.load(scene_gt_info_file)

    for str_im_id in tqdm(scene_gt_dict):
        annos = scene_gt_dict[str_im_id]
        for anno_i, anno in enumerate(annos):
            rotation = np.array(anno["cam_R_m2c"]).reshape(3, 3)
            trans = np.array(anno["cam_t_m2c"]) / 1000
            pose = np.hstack([rotation, trans.reshape(3, 1)])

            ren.clear()
            ren.draw_model(ren_models[0], pose=pose)
            extent = [
                ren_models[0].xsize,
                ren_models[0].ysize,
                ren_models[0].zsize,
            ]
            ren.draw_detection_boundingbox(pose, extent, is_gt=True)
            bgr, depth = ren.finish()
            show_ims = [bgr[:, :, ::-1], depth]
            show_titles = ["color", "depth"]
            grid_show(show_ims, show_titles, row=1, col=2)


if __name__ == "__main__":
    main()
