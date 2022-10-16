import torch
import torch.nn.functional as F
from lietorch import SE3, Sim3

MIN_DEPTH = 0.2

def project(Xs, intrinsics, transxy=None):
    """ Pinhole camera projection """
    assert Xs.ndim == 5, Xs.shape
    X, Y, Z = Xs.unbind(dim=-1)
    fx, fy, cx, cy = intrinsics[:,:,None,None].unbind(dim=-1)

    Z = Z.clamp(min=.01)
    x = fx * (X / Z) + cx
    y = fy * (Y / Z) + cy
    d = 1.0 / Z

    if transxy is not None:
        sc, tx, ty = transxy.unbind(dim=-1)
        x = (x + tx) / sc
        y = (y + ty) / sc

    return torch.stack([x, y, d], dim=-1)


def inv_project(depths, intrinsics, transxy=None, timer=None):
    """ Pinhole camera inverse-projection """
    B, _, H, W = depths.shape
    
    fx, fy, cx, cy = \
        intrinsics[:,:,None,None].unbind(dim=-1)

    if timer is not None:
        timer(f"#1.2.2 {H} {W} {depths.device}")

    # y, x = y_mg, x_mg
    y, x = torch.meshgrid(
        torch.arange(H, device=depths.device).float(), 
        torch.arange(W, device=depths.device).float())

    if timer is not None:
        timer("#1.2.3")

    if transxy is not None:
        sc, tx, ty = transxy.unbind(dim=-1)
        x = x.view(1,1,H,W) * sc - tx
        y = y.view(1,1,H,W) * sc - ty

    X = depths * ((x - cx) / fx)
    Y = depths * ((y - cy) / fy)
    Z = depths

    return torch.stack([X, Y, Z], dim=-1)


def projective_transform_jacobian(Gs, Xs, intrinsics):
    """ Jacobian of projective transform """
    
    X, Y, Z = Xs.unbind(dim=-1)
    fx, fy, cx, cy = \
        intrinsics[:,:,None,None].unbind(dim=-1)

    B, N, H, W = X.shape
    o = torch.zeros_like(Z)
    i = torch.ones_like(Z)
    d = torch.where(Z < 0.2, o, 1.0/Z)

    # projection jacobian (Jp)
    Jp = torch.stack([
        fx*d,    o, -fx*X*d**2,
           o, fy*d, -fy*Y*d**2,
           o,    o,      -d**2,
    ], dim=-1).view(B, N, H, W, 3, 3)

    # action jacobian (Ja)
    if isinstance(Gs, SE3):
        Ja = torch.stack([
            i,  o,  o,  o,  Z, -Y,
            o,  i,  o, -Z,  o,  X, 
            o,  o,  i,  Y, -X,  o,
        ], dim=-1).view(B, N, H, W, 3, 6)

    elif isinstance(Gs, Sim3):
        Ja = torch.stack([
            i,  o,  o,  o,  Z, -Y,  X,
            o,  i,  o, -Z,  o,  X,  Y,
            o,  o,  i,  Y, -X,  o,  Z,
        ], dim=-1).view(B, N, H, W, 3, 7)

    Jj = torch.matmul(Jp, Ja)
    Ji = -Gs[:,:,None,None,None].adjT(Jj)

    return Ji, Jj
    

def projective_transform(poses, depths, intrinsics, ii, jj, jacobian=False, transxy=None, timer=None):
    """ map points from ii->jj """

    Gij = poses[:,jj] * poses[:,ii].inv()

    X0 = inv_project(depths[:,ii], intrinsics[:,ii], transxy=transxy, timer=timer)
    X1 = Gij[:,:,None,None] * X0
    x1 = project(X1, intrinsics[:,jj], transxy=transxy)

    # exclude points too close to camera
    valid = ((X1[...,-1] > MIN_DEPTH) & (X0[...,-1] > MIN_DEPTH)).float()

    if jacobian:
        Ji, Jj = projective_transform_jacobian(Gij, X1, intrinsics[:,jj])
        return x1, valid, (Ji, Jj)

    return x1, valid
