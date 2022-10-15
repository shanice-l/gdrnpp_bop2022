import os.path as osp
import sys
from tqdm import tqdm
import math
import numpy as np
import random
import mmcv

cur_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(cur_dir, "../../../../"))

from lib.pysixd import inout, misc, transform
import ref
from lib.utils.utils import dprint

random.seed(2333)
np.random.seed(2333)


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
    seed = 2333
    scene_dir = osp.join(data_ref.train_dir, f"{scene:06d}")

    N_sample = 20000
    # minNoiseSigma = 0
    # maxNoiseSigma = 15
    # minOutlier = 0
    # maxOutlier = 0.3
else:
    seed = 123
    scene_dir = osp.join(data_ref.test_dir, f"{scene:06d}")
    N_sample = 2000

print("random seed: ", seed)
random.seed(seed)
np.random.seed(seed)
mmcv.mkdir_or_exist(scene_dir)

trans_min = [-2, -2, 4]
trans_max = [2, 2, 8]


def my_rand(a, b):
    return a + (b - a) * random.random()


def random_rotation():
    range = 1

    # use eular formulation, three different rotation angles on 3 axis
    phi = my_rand(0, range * math.pi * 2)
    theta = my_rand(0, range * math.pi)
    psi = my_rand(0, range * math.pi * 2)

    R0 = []
    R0.append(math.cos(psi) * math.cos(phi) - math.cos(theta) * math.sin(phi) * math.sin(psi))
    R0.append(math.cos(psi) * math.sin(phi) + math.cos(theta) * math.cos(phi) * math.sin(psi))
    R0.append(math.sin(psi) * math.sin(theta))

    R1 = []
    R1.append(-(math.sin(psi)) * math.cos(phi) - math.cos(theta) * math.sin(phi) * math.cos(psi))
    R1.append(-(math.sin(psi)) * math.sin(phi) + math.cos(theta) * math.cos(phi) * math.cos(psi))
    R1.append(math.cos(psi) * math.sin(theta))

    R2 = []
    R2.append(math.sin(theta) * math.sin(phi))
    R2.append(-(math.sin(theta)) * math.cos(phi))
    R2.append(math.cos(theta))

    R = []
    R.append(R0)
    R.append(R1)
    R.append(R2)
    return np.array(R)


def main():

    vertex_scale = data_ref.vertex_scale
    obj_id = 1
    model_path = osp.join(model_dir, f"obj_{obj_id:06d}.ply")
    # load the model to calculate bbox
    model = inout.load_ply(model_path, vertex_scale=vertex_scale)

    scene_gt_file = osp.join(scene_dir, "scene_gt.json")
    scene_gt_info_file = osp.join(scene_dir, "scene_gt_info.json")

    scene_gt_dict = {}
    scene_gt_info_dict = {}
    i = 0
    progress_bar = mmcv.ProgressBar(N_sample)
    while True:
        # select grids randomly within the image plane
        # sy = np.random.randint(height, size=N_grid)
        # sx = np.random.randint(width, size=N_grid)
        # rotation = transform.random_rotation_matrix()[:3, :3]
        rotation = random_rotation()
        tx = my_rand(trans_min[0], trans_max[0])
        ty = my_rand(trans_min[1], trans_max[1])
        tz = my_rand(trans_min[2], trans_max[2])
        trans = np.array([tx, ty, tz]).reshape(-1)

        pose = np.hstack([rotation, trans.reshape(3, 1)])
        proj = (K @ trans.T).T
        proj = proj[:2] / proj[2]  # ox, oy

        if proj[0] < 48 or width - proj[0] < 48 or proj[1] < 48 or height - proj[1] < 48:
            dprint(f"skip invalid pose, too close to border, projected center: {proj}")
            continue

        bbox = misc.compute_2d_bbox_xywh_from_pose(model["pts"], pose, K, width=640, height=480, clip=True).tolist()
        x, y, w, h = bbox
        if w < 10 or h < 10:
            dprint(f"skip invalid pose, w: {w}, h: {h}")
            continue

        inst = {
            "cam_R_m2c": rotation.flatten().tolist(),
            "cam_t_m2c": (1000 * trans).flatten().tolist(),  # m to mm
            "obj_id": obj_id,
        }
        scene_gt_dict[str(i)] = [inst]

        info = {"bbox_obj": bbox, "bbox_visib": bbox}
        scene_gt_info_dict[str(i)] = [info]

        i += 1
        progress_bar.update()
        if i >= N_sample:
            break

    inout.save_json(scene_gt_file, scene_gt_dict)
    inout.save_json(scene_gt_info_file, scene_gt_info_dict)
    print(scene_gt_file)
    print(scene_gt_info_file)


if __name__ == "__main__":
    main()
