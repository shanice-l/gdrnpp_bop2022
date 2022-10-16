import torch
import torchvision


def project_points(points_3d, K, TCO):
    assert K.shape[-2:] == (3, 3)
    assert TCO.shape[-2:] == (4, 4)
    batch_size = points_3d.shape[0]
    n_points = points_3d.shape[1]
    device = points_3d.device
    if points_3d.shape[-1] == 3:
        points_3d = torch.cat((points_3d, torch.ones(batch_size, n_points, 1).to(device)), dim=-1)
    P = K @ TCO[:, :3]
    suv = (P.unsqueeze(1) @ points_3d.unsqueeze(-1)).squeeze(-1)
    suv = suv / suv[..., [-1]]
    return suv[..., :2]


def project_points_robust(points_3d, K, TCO, z_min=0.1):
    assert K.shape[-2:] == (3, 3)
    assert TCO.shape[-2:] == (4, 4)
    batch_size = points_3d.shape[0]
    n_points = points_3d.shape[1]
    device = points_3d.device
    if points_3d.shape[-1] == 3:
        points_3d = torch.cat((points_3d, torch.ones(batch_size, n_points, 1).to(device)), dim=-1)
    P = K @ TCO[:, :3]
    suv = (P.unsqueeze(1) @ points_3d.unsqueeze(-1)).squeeze(-1)
    z = suv[..., -1]
    suv[..., -1] = torch.max(torch.ones_like(z) * z_min, z)
    suv = suv / suv[..., [-1]]
    return suv[..., :2]


def boxes_from_uv(uv):
    assert uv.shape[-1] == 2
    x1 = uv[..., [0]].min(dim=1)[0]
    y1 = uv[..., [1]].min(dim=1)[0]

    x2 = uv[..., [0]].max(dim=1)[0]
    y2 = uv[..., [1]].max(dim=1)[0]

    return torch.cat((x1, y1, x2, y2), dim=1)


def crop_to_aspect_ratio(images, box, masks=None, K=None, depths=None):
    assert images.dim() == 4
    bsz, _, h, w = images.shape
    assert box.dim() == 1
    assert box.shape[0] == 4
    w_output, h_output = box[[2, 3]] - box[[0, 1]]
    boxes = torch.cat(
        (torch.arange(bsz).unsqueeze(1).to(box.device).float(), box.unsqueeze(0).repeat(bsz, 1).float()),
        dim=1).to(images.device)
    images = torchvision.ops.roi_pool(images, boxes, output_size=(h_output, w_output))
    if masks is not None:
        assert masks.dim() == 4
        masks = torchvision.ops.roi_pool(masks, boxes, output_size=(h_output, w_output))
    if depths is not None:
        assert depths.dim() == 4, depths.shape
        depths = torchvision.ops.roi_pool(depths, boxes, output_size=(h_output, w_output))
    if K is not None:
        assert K.dim() == 3
        assert K.shape[0] == bsz
        K = get_K_crop_resize(K, boxes[:, 1:], orig_size=(h, w), crop_resize=(h_output, w_output))
    return images, masks, K, depths


def get_K_crop_resize(K, boxes, orig_size, crop_resize):
    """
    Adapted from https://github.com/BerkeleyAutomation/perception/blob/master/perception/camera_intrinsics.py
    Skew is not handled !
    """
    assert K.shape[1:] == (3, 3)
    assert boxes.shape[1:] == (4, )
    K = K.float()
    boxes = boxes.float()
    new_K = K.clone()

    orig_size = torch.as_tensor(orig_size, dtype=torch.float).clone()
    crop_resize = torch.as_tensor(crop_resize, dtype=torch.float).clone()

    final_width, final_height = max(crop_resize), min(crop_resize)
    crop_width = boxes[:, 2] - boxes[:, 0]
    crop_height = boxes[:, 3] - boxes[:, 1]
    crop_cj = (boxes[:, 0] + boxes[:, 2]) / 2
    crop_ci = (boxes[:, 1] + boxes[:, 3]) / 2

    # Crop
    cx = K[:, 0, 2] + (crop_width - 1) / 2 - crop_cj
    cy = K[:, 1, 2] + (crop_height - 1) / 2 - crop_ci

    # # Resize (upsample)
    center_x = (crop_width - 1) / 2
    center_y = (crop_height - 1) / 2
    orig_cx_diff = cx - center_x
    orig_cy_diff = cy - center_y
    scale_x = final_width / crop_width
    scale_y = final_height / crop_height
    scaled_center_x = (final_width - 1) / 2
    scaled_center_y = (final_height - 1) / 2
    fx = scale_x * K[:, 0, 0]
    fy = scale_y * K[:, 1, 1]
    cx = scaled_center_x + scale_x * orig_cx_diff
    cy = scaled_center_y + scale_y * orig_cy_diff

    new_K[:, 0, 0] = fx
    new_K[:, 1, 1] = fy
    new_K[:, 0, 2] = cx
    new_K[:, 1, 2] = cy
    return new_K
