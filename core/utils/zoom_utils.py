import torch
from detectron2.layers.roi_align import ROIAlign
from torchvision.ops import RoIPool


def deepim_boxes(
    ren_boxes,
    ren_centers_2d,
    obs_boxes=None,
    lamb=1.4,
    imHW=(480, 640),
    outHW=(480, 640),
    clamp=False,
):
    """
    Args:
        ren_boxes: N x 4
        ren_centers_2d: Nx2, rendered object center is the crop center
        obs_boxes: N x 4, if None, only use the rendered boxes/centers to determine the crop region
        lamb: enlarge the scale of cropped region
        imH (int):
        imW (int):
    Returns:
        crop_boxes (Tensor): N x 4, either the common region from obs/ren or just obs
        resize_ratios (Tensor): Nx2, resize ratio of (w,h), actually the same in w,h because we keep the aspect ratio
    """
    ren_x1, ren_y1, ren_x2, ren_y2 = (ren_boxes[:, i] for i in range(4))  # (N,)
    ren_cx = ren_centers_2d[:, 0]  # (N,)
    ren_cy = ren_centers_2d[:, 1]  # (N,)

    outH, outW = outHW
    aspect_ratio = outW / outH  # 4/3 or 1

    if obs_boxes is not None:
        obs_x1, obs_y1, obs_x2, obs_y2 = (obs_boxes[:, i] for i in range(4))  # (N,)
        xdists = torch.stack(
            [
                ren_cx - obs_x1,
                ren_cx - ren_x1,
                obs_x2 - ren_cx,
                ren_x2 - ren_cx,
            ],
            dim=1,
        ).abs()
        ydists = torch.stack(
            [
                ren_cy - obs_y1,
                ren_cy - ren_y1,
                obs_y2 - ren_cy,
                ren_y2 - ren_cy,
            ],
            dim=1,
        ).abs()
    else:
        xdists = torch.stack([ren_cx - ren_x1, ren_x2 - ren_cx], dim=1).abs()
        ydists = torch.stack([ren_cy - ren_y1, ren_y2 - ren_cy], dim=1).abs()
    xdist = xdists.max(dim=1)[0]  # (N,)
    ydist = ydists.max(dim=1)[0]

    crop_h = torch.max(xdist / aspect_ratio, ydist).clamp(min=1) * 2 * lamb  # (N,)
    crop_w = crop_h * aspect_ratio  # (N,)

    x1, y1, x2, y2 = (
        ren_cx - crop_w / 2,
        ren_cy - crop_h / 2,
        ren_cx + crop_w / 2,
        ren_cy + crop_h / 2,
    )
    boxes = torch.stack([x1, y1, x2, y2], dim=1)
    assert not clamp
    if clamp:
        imH, imW = imHW
        boxes[:, [0, 2]] = torch.clamp(boxes[:, [0, 2]], 0, imW - 1)
        boxes[:, [1, 3]] = torch.clamp(boxes[:, [1, 3]], 0, imH - 1)

    resize_ratios = torch.stack([outW / crop_w, outH / crop_h], dim=1)
    return boxes, resize_ratios


def batch_crop_resize(x, rois, out_H, out_W, aligned=True, interpolation="bilinear"):
    """
    Args:
        x: BCHW
        rois: Bx5, rois[:, 0] is the idx into x
        out_H (int):
        out_W (int):
    """
    output_size = (out_H, out_W)
    if interpolation == "bilinear":
        op = ROIAlign(output_size, 1.0, 0, aligned=aligned)
    elif interpolation == "nearest":
        op = RoIPool(output_size, 1.0)  #
    else:
        raise ValueError(f"Wrong interpolation type: {interpolation}")
    return op(x, rois)
