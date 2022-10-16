import torch

from .transforms import rad2deg, transform_pts

def _transform_pts_using_sym(TXO_pred, TXO_gt, points, symmetries):
    N_SYM = symmetries.shape[1]
    P_bar_S = (TXO_pred.unsqueeze(1) @ symmetries).flatten(0,1)
    P_hat = TXO_gt.repeat_interleave(N_SYM, dim=0)
    points_expanded = points.repeat_interleave(N_SYM, dim=0)
    assert P_hat.shape == P_bar_S.shape and P_bar_S.ndim == 3
    P_bar_S_points = transform_pts(P_bar_S, points_expanded)
    P_hat_points = transform_pts(P_hat, points_expanded)
    return P_bar_S_points, P_hat_points

def calc_mssd_recall(TXO_pred, TXO_gt, points, symmetries, diameters):
    DEVICE, (B, N_SYM, _, _) = symmetries.device, symmetries.shape
    P_bar_S_points, P_hat_points = _transform_pts_using_sym(TXO_pred, TXO_gt, points, symmetries)
    l2_dist = (P_hat_points - P_bar_S_points).pow(2).sum(dim=-1).sqrt()
    l2_dist = l2_dist.view(B, N_SYM, -1)
    mssd = l2_dist.max(dim=-1).values.min(dim=-1).values # B
    threshholds = diameters.unsqueeze(1) * torch.linspace(.05, .5, 10, device=DEVICE).unsqueeze(0)
    mssd_recall = (mssd.unsqueeze(1) < threshholds).float().mean(dim=1)
    return mssd_recall

def _my_project_pts(K, pts):
    assert K.shape[0] == pts.shape[0], [K.shape, pts.shape]
    assert K.shape[1] == K.shape[2] == 3
    fx, fy, cx, cy = K[:,0,0,None], K[:,1,1,None], K[:,0,2,None], K[:,1,2,None] # 64
    px, py, pz = pts.unbind(dim=-1)
    ix = (px*fx)/pz + cx
    iy = (py*fy)/pz + cy
    return torch.stack((ix,iy), dim=-1)

def calc_mspd_recall(TXO_pred, TXO_gt, points, symmetries, K):
    DEVICE, (B, N_SYM, _, _) = symmetries.device, symmetries.shape
    P_bar_S_points, P_hat_points = _transform_pts_using_sym(TXO_pred, TXO_gt, points, symmetries)
    P_bar_S_points_proj = _my_project_pts(K.repeat_interleave(N_SYM, dim=0), P_bar_S_points)
    P_hat_points_proj = _my_project_pts(K.repeat_interleave(N_SYM, dim=0), P_hat_points)
    l2_dist = (P_hat_points_proj - P_bar_S_points_proj).pow(2).sum(dim=-1).sqrt()
    l2_dist = l2_dist.view(B, N_SYM, -1)
    mspd = l2_dist.max(dim=-1).values.min(dim=-1).values # B
    threshholds = torch.linspace(5, 50, 10, device=DEVICE).unsqueeze(0) # B x 10
    mspd_recall = (mspd.unsqueeze(1) < threshholds).float().mean(dim=1)
    return mspd_recall

def calc_rot_error(TXO_pred, TXO_gt, symmetries):
    P_bar_S = (TXO_pred.unsqueeze(1) @ symmetries)[...,:3,:3]
    P_hat = TXO_gt[:,None,:3,:3]
    relative_pose = P_hat @ torch.inverse(P_bar_S) # b N 3 3
    trace = torch.diagonal(relative_pose, dim1=-2, dim2=-1).sum(-1).clamp(min=-1, max=3)
    assert trace.ndim == 2
    angle_error_degrees = rad2deg(torch.acos((trace-1)/2))
    assert not torch.isnan(angle_error_degrees).any()
    return angle_error_degrees.min(dim=1).values
