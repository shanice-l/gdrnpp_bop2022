import warnings

import gin
import numpy as np
import torch
from lietorch import SE3, SO3, Sim3
from scipy.spatial.transform import Rotation as R

from .geom import projective_ops_rgbd as pops
from .transforms import mat2SE3, SE3_stack

def vectorize_intrinsics(K_batch):
    return K_batch[...,[0,1,0,1],[0,1,2,2]]

def get_ind(N, DEVICE):
    ii, jj = torch.arange(N, device=DEVICE), torch.full((N,), N, device=DEVICE)
    return torch.cat((ii, jj)), torch.cat((jj, ii))

def pose_metrics(dE):
    """ Translation/Rotation/Scaling metrics from Sim3 """

    t, q, s = dE.data.split([3, 4, 1], -1)
    q = q.contiguous()
    ang = SO3(q).log().norm(dim=-1)

    r_err = (180/np.pi) * ang
    t_err = t.norm(dim=-1)
    s_err = (s - 1.0).abs()

    return r_err, t_err, s_err

@gin.configurable
def geodesic_and_flow_loss(model_out, possible_gt, solver_method, intrinsics_mat, labels, N, renderer, ITER_GAMMA, GEODESIC_WEIGHT, TARGET_WEIGHT):
    """ Loss function for training network """

    GT_64poses = mat2SE3(possible_gt)
    model_out['Gs'] = [SE3(e) for e in model_out['Gs']]
    predicted_poses = model_out['Gs'][-1][:, [-1]]
    geodesic_dists = (predicted_poses * GT_64poses.inv()).log()
    distances = geodesic_dists.abs().norm(dim=-1)
    min_dist_idxs = distances.argmin(dim=1)
    GT_poses = SE3_stack([p64[minidx.item()] for minidx, p64 in zip(min_dist_idxs, GT_64poses)], dim=0)[:, None]

    raft_iters = len(model_out['coords'])
    ii, jj = get_ind(N, possible_gt.device)
    adjusted_loss_gamma = ITER_GAMMA**(9/(raft_iters - 1)) if raft_iters > 1 else ITER_GAMMA # accounting for the number of inner loops
    iter_weights = np.array([adjusted_loss_gamma ** (raft_iters - i - 1) for i in range(raft_iters)])
    iter_weights = iter_weights * (raft_iters/iter_weights.sum())
    target_loss = []

    depths_gt = model_out["depths"][0].clone()
    if solver_method == "Modified BD-PnP":
        with torch.no_grad():
            B, _, lowres_H, lowres_W = depths_gt.shape
            res_rep = torch.tensor([(lowres_H, lowres_W)]*B).to(device=depths_gt.device).mul(4)
            _, gt_photo_depth, _ = renderer(labels, GT_poses[:, -1].matrix(), intrinsics_mat[:,-1], res_rep, scale_res=(1/4))
        depths_gt[:,-1] = gt_photo_depth.clamp(min=1e-3)

    for i in range(raft_iters):
        if not model_out["masks"][i][:,ii].any():
            target_loss.append(torch.tensor(0, device=depths_gt.device))
            warnings.warn("There are no valid pixels in the output. Setting loss to 0.")
            break
        GT_Gs = SE3(model_out['Gs'][i].data.detach().clone())
        assert GT_Gs[:, [-1]].shape == GT_poses.shape
        GT_Gs[:, [-1]] = GT_poses

        intrinsics = vectorize_intrinsics(intrinsics_mat) / 4
        coords_gt, valid = pops.projective_transform(GT_Gs, depths_gt, intrinsics, ii, jj)
        assert valid.bool().any()
        assert (solver_method == "BD-PnP") or gt_photo_depth[valid[:,-1].bool()].min() > 2e-3
        assert valid.shape == model_out["masks"][i][:,ii].shape

        valid = valid.bool() & model_out["masks"][i][:,ii]

        # Target loss
        assert valid.any()
        assert model_out["weights"][i].shape == coords_gt.shape

        target_err = (coords_gt - model_out['coords'][i])[..., :2][valid].abs()
        target_loss.append( iter_weights[i] * target_err.mean() )

    geodesic_loss = []
    for i in range(raft_iters):
        geodesic_dists = (model_out['Gs'][i][:, [-1]] * GT_poses.inv()).log()
        geodesic_loss.append( iter_weights[i] * geodesic_dists.abs().norm(dim=-1).mean() )

    dE = Sim3(model_out['Gs'][i][:, [-1]] * GT_poses.inv()).detach() # 4x1
    r_err, _, _ = pose_metrics(dE)
    assert not torch.isnan(r_err).any() and r_err.shape == dE.shape

    metrics = {
        'target_loss': torch.stack(target_loss) * TARGET_WEIGHT,
        'geodesic_loss': torch.stack(geodesic_loss) * GEODESIC_WEIGHT,
    }

    loss = GEODESIC_WEIGHT * sum(geodesic_loss) + TARGET_WEIGHT * sum(target_loss)

    return loss.unsqueeze(0), metrics
