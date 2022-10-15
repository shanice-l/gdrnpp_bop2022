import numpy as np
import cv2
import torch


def get_K_crop_resize(K, crop_xy, resize_ratio):
    """
    Args:
        K: [b,3,3]
        crop_xy: [b, 2]  left top of crop boxes
        resize_ratio: [b,2] or [b,1]
    """
    assert K.shape[1:] == (3, 3)
    assert crop_xy.shape[1:] == (2,)
    assert resize_ratio.shape[1:] == (2,) or resize_ratio.shape[1:] == (1,)
    bs = K.shape[0]

    new_K = K.clone()
    new_K[:, [0, 1], 2] = K[:, [0, 1], 2] - crop_xy  # [b, 2]
    new_K[:, [0, 1]] = new_K[:, [0, 1]] * resize_ratio.view(bs, -1, 1)
    return new_K


def project_points(points_3d, K, pose, z_min=None):
    """
    Args:
        points_3d: BxPx3
        K: Bx3x3
        pose: Bx3x4
        z_min: prevent zero devision, eg. 0.1
    Returns:
        projected 2d points: BxPx2
    """
    assert K.shape[-2:] == (3, 3)
    assert pose.shape[-2:] == (3, 4)
    batch_size = points_3d.shape[0]
    n_points = points_3d.shape[1]
    device = points_3d.device
    if points_3d.shape[-1] == 3:
        points_3d = torch.cat((points_3d, torch.ones(batch_size, n_points, 1).to(device)), dim=-1)
    P = K @ pose[:, :3]
    suv = (P.unsqueeze(1) @ points_3d.unsqueeze(-1)).squeeze(-1)  # Bx1x3x4 @ BxPx4x1 -> BxPx3
    if z_min is not None:
        z = suv[..., -1]
        suv[..., -1] = torch.max(torch.ones_like(z) * z_min, z)  # eg. z_min=0.1
    suv = suv / suv[..., [-1]]
    return suv[..., :2]  # BxPx2


def centers_2d_from_t(K, t, z_min=None):
    """can also get the centers via projecting the zero point (B,1,3)
    Args:
        K: Bx3x3
        t: Bx3
        z_min: to prevent zero division
    Returns:
        centers_2d: Bx2
    """
    assert K.ndim == 3 and K.shape[-2:] == (3, 3), K.shape
    bs = K.shape[0]
    proj = (K @ t.view(bs, 3, 1)).view(bs, 3)
    if z_min is not None:
        z = proj[..., -1]
        proj[..., -1] = torch.max(torch.ones_like(z) * z_min, z)  # eg. z_min=0.1
    centers_2d = proj[:, :2] / proj[:, [-1]]  # Nx2
    return centers_2d


def centers_2d_from_pose(K, pose, z_min=None):
    """can also get the centers via projecting the zero point (B,1,3)
    Args:
        K: Bx3x3
        pose: Bx3x4 (only use the transltion)
        z_min: to prevent zero division
    Returns:
        centers_2d: Bx2
    """
    assert K.ndim == 3 and K.shape[-2:] == (3, 3), K.shape
    assert pose.ndim == 3 and pose.shape[-2:] == (3, 4), pose.shape
    bs = pose.shape[0]
    proj = (K @ pose[:, :3, [3]]).view(bs, 3)  # Nx3x3 @ Nx3x1 -> Nx3x1 -> Nx3
    if z_min is not None:
        z = proj[..., -1]
        proj[..., -1] = torch.max(torch.ones_like(z) * z_min, z)  # eg. z_min=0.1
    centers_2d = proj[:, :2] / proj[:, [-1]]  # Nx2
    return centers_2d


def boxes_from_points_2d(uv):
    """
    Args:
        uv: BxPx2 projected 2d points
    Returns:
        Bx4
    """
    assert uv.ndim == 3 and uv.shape[-1] == 2, uv.shape
    x1 = uv[..., 0].min(dim=1)[0]  # (B,)
    y1 = uv[..., 1].min(dim=1)[0]

    x2 = uv[..., 0].max(dim=1)[0]
    y2 = uv[..., 1].max(dim=1)[0]

    return torch.stack([x1, y1, x2, y2], dim=1)  # Bx4


def bboxes_from_pose(points_3d, K, pose, z_min=None, imH=480, imW=640, clamp=False):
    points_2d = project_points(points_3d, K=K, pose=pose, z_min=z_min)
    boxes = boxes_from_points_2d(points_2d)
    if clamp:
        boxes[:, [0, 2]] = torch.clamp(boxes[:, [0, 2]], 0, imW - 1)
        boxes[:, [1, 3]] = torch.clamp(boxes[:, [1, 3]], 0, imH - 1)
    return boxes


def adapt_image_by_K(
    image, *, K_old, K_new, interpolation=cv2.INTER_LINEAR, border_type=cv2.BORDER_REFLECT, height=480, width=640
):
    """adapt image from old K to new K."""
    H_old, W_old = image.shape[:2]
    K_old = K_old.copy()
    K_old[0, :] = K_old[0, :] / W_old * width
    K_old[1, :] = K_old[1, :] / H_old * height

    focal_scale_x = K_new[0, 0] / K_old[0, 0]
    focal_scale_y = K_new[1, 1] / K_old[1, 1]
    ox, oy = K_new[0, 2] - K_old[0, 2], K_new[1, 2] - K_old[1, 2]

    image = cv2.resize(
        image,
        (int(width * focal_scale_x), int(height * focal_scale_y)),
        interpolation=interpolation,
    )

    image = cv2.copyMakeBorder(image, 200, 200, 200, 200, borderType=border_type)
    # print(image.shape)
    y1 = int(round(image.shape[0] / 2 - oy - height / 2))
    y2 = int(round(image.shape[0] / 2 - oy + height / 2))
    x1 = int(round(image.shape[1] / 2 - ox - width / 2))
    x2 = int(round(image.shape[1] / 2 - ox + width / 2))
    # print(x1, y1, x2, y2)
    return image[y1:y2, x1:x2]
