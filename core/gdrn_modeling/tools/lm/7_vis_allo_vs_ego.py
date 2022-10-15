import mmcv
import os.path as osp
import numpy as np
import sys
from transforms3d.axangles import axangle2mat
import torch

cur_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(cur_dir, "../../../../"))

from core.utils.utils import allocentric_to_egocentric
from lib.egl_renderer.egl_renderer_v3 import EGLRenderer

model_dir = "datasets/BOP_DATASETS/lm/models/"

obj_id = 2
model_paths = [osp.join(model_dir, f"obj_{obj_id:06d}.ply")]
texture_paths = None  # [osp.join(model_dir, f"obj_{obj_id:06d}.png")]

height = 480
width = 640
# K = np.array([[1066.778, 0.0, 312.9869], [0.0, 1067.487, 241.3109], [0.0, 0.0, 1.0]])
K = np.array([[572.4114, 0, 325.2611], [0, 573.57043, 242.04899], [0, 0, 1]])
ren = EGLRenderer(
    model_paths,
    vertex_scale=0.001,
    use_cache=True,
    width=width,
    height=height,
    K=K,
    texture_paths=texture_paths,
)


tensor_kwargs = {"device": torch.device("cuda"), "dtype": torch.float32}
image_tensor = torch.empty((height, width, 4), **tensor_kwargs).detach()


R1 = axangle2mat((1, 0, 0), angle=0.5 * np.pi)
R2 = axangle2mat((0, 0, 1), angle=-0.5 * np.pi)
R = np.dot(R1, R2)
z = 0.7
t = np.array([0, 0, z], dtype=np.float32)
t1 = np.array([-0.25, 0, z], dtype=np.float32)
t2 = np.array([0.25, 0, z], dtype=np.float32)
pose = np.hstack([R, t.reshape((3, 1))])
pose1 = np.hstack([R, t1.reshape((3, 1))])
pose2 = np.hstack([R, t2.reshape((3, 1))])

cam_ray = (0, 0, 1)
allo_pose = allocentric_to_egocentric(pose, cam_ray=cam_ray)
allo_pose1 = allocentric_to_egocentric(pose1, cam_ray=cam_ray)
allo_pose2 = allocentric_to_egocentric(pose2, cam_ray=cam_ray)

background = np.ones((height, width, 3), dtype="uint8") * 255

ren.render(
    [0, 0, 0],
    [pose1, pose, pose2],
    image_tensor=image_tensor,
    background=background,
)
im_ego = (image_tensor[:, :, :3].detach().cpu().numpy() + 0.5).astype("uint8")
# grid_show([im_ego[:,:,::-1]], ["im_ego"], row=1, col=1)

# ren.render([0], [pose1], image_tensor=image_tensor)
# im_ego1 = (image_tensor[:, :, :3].detach().cpu().numpy() + 0.5).astype('uint8')
#
# ren.render([0], [pose2], image_tensor=image_tensor)
# im_ego2 = (image_tensor[:, :, :3].detach().cpu().numpy() + 0.5).astype('uint8')


ren.render(
    [0, 0, 0],
    [allo_pose1, allo_pose, allo_pose2],
    image_tensor=image_tensor,
    background=background,
)
im_allo = (image_tensor[:, :, :3].detach().cpu().numpy() + 0.5).astype("uint8")
# ren.render([0], [allo_pose1], image_tensor=image_tensor)
# im_allo1 = (image_tensor[:, :, :3].detach().cpu().numpy() + 0.5).astype('uint8')
# ren.render([0], [allo_pose2], image_tensor=image_tensor)
# im_allo2 = (image_tensor[:, :, :3].detach().cpu().numpy() + 0.5).astype('uint8')

# grid_show([im_ego1[:, :, ::-1], im_ego[:, :, ::-1], im_ego2[:, :, ::-1],
#            im_allo1[:, :, ::-1], im_allo[:, :, ::-1], im_allo2[:, :, ::-1]],
#           ["ego1", "ego", "ego2",
#            "allo1", "allo", "allo2"],
#           row=2, col=3)
# grid_show([im_ego[:, :, ::-1],  im_allo[:, :, ::-1]],
#           ["ego",
#            "allo"],
#           row=1, col=2)

mmcv.imwrite(im_ego, "output/ego.png")
mmcv.imwrite(im_allo, "output/allo.png")
