import gin
import torch
import torch.nn.functional as F
from torch_scatter import scatter_sum

from .chol import schur_solve
from .projective_ops_rgb import \
    projective_transform as projective_transform_rgb
from .projective_ops_rgbd import \
    projective_transform as projective_transform_rgbd

"""
Modified BD-PnP Solver
"""

# utility functions for scattering ops
def safe_scatter_add_mat(A, ii, jj, n, m):
    v = (ii >= 0) & (jj >= 0) & (ii < n) & (jj < m)
    return scatter_sum(A[:,v], ii[v]*m + jj[v], dim=1, dim_size=n*m)

def safe_scatter_add_vec(b, ii, n):
    v = (ii >= 0) & (ii < n)
    return scatter_sum(b[:,v], ii[v], dim=1, dim_size=n)

# apply retraction operator to inv-depth maps
def disp_retr(disps, dz, ii):
    ii = ii.to(device=dz.device)
    return disps + scatter_sum(dz, ii, dim=1, dim_size=disps.shape[1])

# apply retraction operator to poses
def pose_retr(poses, dx, ii):
    ii = ii.to(device=dx.device)
    return poses.retr(scatter_sum(dx, ii, dim=1, dim_size=poses.shape[1]))


@gin.configurable()
def BA(target, weight, eta, poses, disps, intrinsics, ii, jj, motion_only=False):
    """ Full Bundle Adjustment """

    fixedp = fixedd = (poses.shape[1]-1)
    M=1

    B, P, ht, wd = disps.shape
    N = ii.shape[0]
    D = poses.manifold_dim

    ### 1: commpute jacobians and residuals ###
    coords, valid, (Ji, Jj, Jz) = projective_transform_rgb(
        poses, disps, intrinsics, ii, jj, jacobian=True, return_depth=False)
    target = target[..., :2]
    weight = weight[..., :2]

    r = (target - coords).view(B, N, -1, 1)
    w_scale = 0.001
    w = w_scale * (valid * weight).view(B, N, -1, 1)

    ### 2: construct linear system ###
    Ji = Ji.reshape(B, N, -1, D)
    Jj = Jj.reshape(B, N, -1, D)
    wJiT = (w * Ji).transpose(2,3)
    wJjT = (w * Jj).transpose(2,3)

    Jz = Jz.reshape(B, N, ht*wd, -1)

    Hii = torch.matmul(wJiT, Ji)
    Hij = torch.matmul(wJiT, Jj)
    Hji = torch.matmul(wJjT, Ji)
    Hjj = torch.matmul(wJjT, Jj)

    vi = torch.matmul(wJiT, r).squeeze(-1)
    vj = torch.matmul(wJjT, r).squeeze(-1)

    Ei = (wJiT.view(B,N,D,ht*wd,-1) * Jz[:,:,None]).sum(dim=-1)
    Ej = (wJjT.view(B,N,D,ht*wd,-1) * Jz[:,:,None]).sum(dim=-1)

    w = w.view(B, N, ht*wd, -1)
    r = r.view(B, N, ht*wd, -1)
    wk = torch.sum(w*r*Jz, dim=-1)
    Ck = torch.sum(w*Jz*Jz, dim=-1)

    kk = ii.clone()
    kk = kk-fixedd
    
    # only optimize keyframe poses
    P = 1
    ii = ii - fixedp
    jj = jj - fixedp

    H = safe_scatter_add_mat(Hii, ii, ii, P, P) + \
        safe_scatter_add_mat(Hij, ii, jj, P, P) + \
        safe_scatter_add_mat(Hji, jj, ii, P, P) + \
        safe_scatter_add_mat(Hjj, jj, jj, P, P)

    E = safe_scatter_add_mat(Ei, ii, kk, P, M) + \
        safe_scatter_add_mat(Ej, jj, kk, P, M)

    v = safe_scatter_add_vec(vi, ii, P) + \
        safe_scatter_add_vec(vj, jj, P)

    C = safe_scatter_add_vec(Ck, kk, M)
    w = safe_scatter_add_vec(wk, kk, M)

    C = C + eta.view(*C.shape) + 1e-7 # This is C from Ceres pdf: http://ceres-solver.org/nnls_solving.html?highlight=normal%20equations#equation-hblock

    H = H.view(B, P, P, D, D) # This is B from Ceres pdf.
    E = E.view(B, P, M, D, ht*wd)

    ### 3: solve the system ###
    if motion_only:
        E = torch.zeros_like(E)
    dx, dz = schur_solve(H, E, C, v, w, ep=0.1, lm=0.0001)
    ### 4: apply retraction ###
    poses = pose_retr(poses, dx, torch.arange(P) + fixedp)
    disps = disp_retr(disps, dz.view(B,-1,ht,wd), torch.arange(M) + fixedd)
    disps = torch.where(disps > 10, torch.zeros_like(disps), disps)
    disps = disps.clamp(min=0.0)
    return poses, disps


"""
BD-PnP Solver
"""


class DenseSystemSolver(torch.autograd.Function):
    @staticmethod
    def forward(ctx, H, b):
        U = torch.cholesky(H)

        xs = torch.cholesky_solve(b, U)
        ctx.save_for_backward(U, xs)
        return xs

    @staticmethod
    def backward(ctx, grad_x):
        grad_x = torch.where(torch.isnan(grad_x), 
            torch.zeros_like(grad_x), grad_x)

        U, xs = ctx.saved_tensors
        dz = torch.cholesky_solve(grad_x, U)
        dH = -torch.matmul(xs, dz.transpose(-1,-2))

        return dH, dz


def _linearize_moba(target, weight, poses, depths, intrinsics, ii, jj):
    bdim, mdim = B, M = poses.shape[:2]
    ddim = D = poses.manifold_dim
    ndim = N = ii.shape[0]
    ### 1: commpute jacobians and residuals ###
    coords, val, (Ji, Jj) = projective_transform_rgbd(
        poses, depths, intrinsics, ii, jj, jacobian=True)
    val = val * (depths[:, ii] > 0.1).float()
    val = val.unsqueeze(-1)
    r = (target - coords).view(B, N, -1, 1)
    w = (val * weight).view(B, N, -1, 1)
    ### 2: construct linear system ###
    Ji = Ji.view(B, N, -1, 6)
    Jj = Jj.view(B, N, -1, 6)
    wJiT = (.001 * w * Ji).transpose(2, 3)
    wJjT = (.001 * w * Jj).transpose(2, 3)
    Hii = torch.matmul(wJiT, Ji)
    Hij = torch.matmul(wJiT, Jj)
    Hji = torch.matmul(wJjT, Ji)
    Hjj = torch.matmul(wJjT, Jj)
    vi = torch.matmul(wJiT, r).squeeze(-1)
    vj = torch.matmul(wJjT, r).squeeze(-1)

    device = Jj.device

    H = torch.zeros(bdim, mdim*mdim, ddim, ddim, device=device)
    v = torch.zeros(bdim, mdim, ddim, device=device)

    H.scatter_add_(1, (ii*mdim + ii).view(1,ndim,1,1).repeat(bdim,1,ddim,ddim), Hii)
    H.scatter_add_(1, (ii*mdim + jj).view(1,ndim,1,1).repeat(bdim,1,ddim,ddim), Hij)
    H.scatter_add_(1, (jj*mdim + ii).view(1,ndim,1,1).repeat(bdim,1,ddim,ddim), Hji)
    H.scatter_add_(1, (jj*mdim + jj).view(1,ndim,1,1).repeat(bdim,1,ddim,ddim), Hjj)
    H = H.view(bdim, mdim, mdim, ddim, ddim)

    v.scatter_add_(1, ii.view(1,ndim,1).repeat(bdim,1,ddim), vi)
    v.scatter_add_(1, jj.view(1,ndim,1).repeat(bdim,1,ddim), vj)

    return H, v

def _step_moba(target, weight, poses, depths, intrinsics, ii, jj, ep_lmbda=10.0, lm_lmbda=0.00001):

    bd, kd, ht, wd = depths.shape
    md = poses.shape[1]
    D = poses.manifold_dim

    H, v = _linearize_moba(target, weight, poses, depths, intrinsics, ii, jj)

    H = H.permute(0, 1, 3, 2, 4).reshape(bd, D*md, D*md)
    v = v.reshape(bd, D*md, 1)

    dI = torch.eye(D*md, device=H.device)
    _H = H + ep_lmbda*dI + lm_lmbda*H*dI

    # fix first pose
    _H = _H[:, -D:, -D:]
    _v = v[:, -D:]

    dx = DenseSystemSolver.apply(_H, _v)
    dx = dx.view(bd, 1, D)
    # dx = dx.clamp(-2.0, 2.0)

    fill = torch.zeros_like(dx[:,:1].repeat(1,md-1,1))
    dx = torch.cat([fill, dx], dim=1)

    poses = poses.retr(dx)

    return poses, depths, intrinsics

def MoBA(target, weight, poses, depths, intrinsics, num_steps, ii, jj):
    """ Motion only bundle adjustment """
    for itr in range(num_steps):
        poses, depths, intrinsics = _step_moba(target, weight, poses, depths, intrinsics, ii, jj)

    return poses, depths, intrinsics
