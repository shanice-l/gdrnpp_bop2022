from itertools import chain, permutations, product

import gin
import numpy as np
import torch
import transforms3d
from lietorch import SE3
from scipy.spatial.transform import Rotation as R
from matplotlib import pyplot as plt

gen = torch.Generator().manual_seed

det = lambda t: np.linalg.det(t.detach().cpu().numpy())

transform_pts = lambda T, pts: (mat2SE3(T.unsqueeze(1)) * pts)

rad2deg = lambda t: ((180/np.pi)*t)
deg2rad = lambda t: ((np.pi/180)*t)

# From Cosypose
@gin.configurable
def add_noise(TCO, euler_deg_std, trans_std, generator):
    TCO_out = TCO.clone()
    device = TCO_out.device
    bsz = TCO.shape[0]
    euler_noise_deg = np.concatenate(
        [torch.normal(mean=torch.zeros(bsz, 1), std=torch.full((bsz, 1), euler_deg_std_i), generator=generator).numpy()
         for euler_deg_std_i in euler_deg_std], axis=1)
    euler_noise_rad = deg2rad(euler_noise_deg)
    R_noise = torch.tensor([transforms3d.euler.euler2mat(*xyz) for xyz in euler_noise_rad]).float().to(device)

    trans_noise = np.concatenate(
        [torch.normal(mean=torch.zeros(bsz, 1), std=torch.full((bsz, 1), trans_std_i), generator=generator).numpy()
         for trans_std_i in trans_std], axis=1)
    trans_noise = torch.tensor(trans_noise).float().to(device)
    TCO_out[:, :3, :3] = TCO_out[:, :3, :3] @ R_noise
    TCO_out[:, :3, 3] += trans_noise
    return TCO_out

@gin.configurable
def get_perturbations(TCO: torch.Tensor, extra_views=False) -> torch.Tensor:
    params = product(permutations("XYZ", 1), [-45/2,45/2])
    if extra_views:
        params = chain(params, product(permutations("XYZ", 1), [-45,45]))
    perturbations = torch.eye(4, device=TCO.device).tile(1, 7+extra_views*6, 1, 1)
    perturbations[0, 1:, :3, :3] = torch.stack([torch.from_numpy(R.from_euler(''.join(a), s, degrees=True).as_matrix()) for a, s in params], dim=0)
    return TCO.unsqueeze(1) @ perturbations

@torch.no_grad()
def mat2quat(mat):
    assert mat.shape[-2] == mat.shape[-1] == 3
    assert det(mat).min() > 0.95 and det(mat).max() < 1.05, det(mat)
    *shape, _, _ = mat.shape
    mat_list = list(mat.view(-1,3,3).detach().cpu().numpy())
    quat_np = np.stack([R.from_matrix(m).as_quat() for m in mat_list], axis=0)
    return torch.from_numpy(quat_np).to(mat).view(*shape, 4)

@torch.no_grad()
def mat2SE3(mat: torch.Tensor) -> SE3:
    assert mat.shape[-2] == mat.shape[-1] == 4
    quat = mat2quat(mat[...,:3,:3])
    trans = mat[..., :3, 3]
    assert trans.ndim == quat.ndim, (trans.shape, quat.shape)
    data = torch.cat((trans, quat), dim=-1)
    return SE3(data)

def SE3_stack(iterable, dim):
    assert dim >= 0
    return SE3(torch.stack([e.data for e in iterable], dim=dim))

def depth_to_jet(depth, scale_vmin=1.0, max_val=np.inf):
    valid = (depth > 1e-2) & (depth < max_val)
    vmin = depth[valid].min() * scale_vmin
    vmax = depth.max()
    cmap = plt.cm.jet
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    depth = cmap(norm(depth))
    depth[~valid] = 1
    return np.ascontiguousarray(depth[...,:3] * 255, dtype=np.uint8)