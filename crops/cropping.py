import gin
import torch
import torchvision

from .camera_geometry import (boxes_from_uv, get_K_crop_resize, project_points,
                              project_points_robust)


@gin.configurable
def crop_inputs(renderer, images, K, TCO, labels, masks, sce_depth, render_size):
    bsz, nchannels, h, w = images.shape
    assert K.shape == (bsz, 3, 3)
    assert TCO.shape == (bsz, 4, 4)
    assert len(labels) == bsz
    points = renderer.get_pointclouds(labels)
    uv = project_points(points, K, TCO)
    boxes_rend = boxes_from_uv(uv)
    boxes_crop, images_cropped, masks_cropped, sce_depth_cropped = deepim_crops_robust(
        images=images, obs_boxes=boxes_rend, K=K,
        TCO_pred=TCO, O_vertices=points, output_size=render_size, lamb=1.4, masks=masks, sce_depth=sce_depth,
    )
    K_crop = get_K_crop_resize(K=K.clone(), boxes=boxes_crop,
                                orig_size=images.shape[-2:], crop_resize=render_size)

    return images_cropped, K_crop.detach(), boxes_rend, boxes_crop, masks_cropped, sce_depth_cropped

def deepim_boxes(rend_center_uv, obs_boxes, rend_boxes, lamb=1.4, im_size=(240, 320), clamp=False):
    """
    gt_boxes: N x 4
    crop_boxes: N x 4
    """
    lobs, robs, uobs, dobs = obs_boxes[:, [0, 2, 1, 3]].t()
    lrend, rrend, urend, drend = rend_boxes[:, [0, 2, 1, 3]].t()
    xc = rend_center_uv[..., 0, 0]
    yc = rend_center_uv[..., 0, 1]
    lobs, robs = lobs.unsqueeze(-1), robs.unsqueeze(-1)
    uobs, dobs = uobs.unsqueeze(-1), dobs.unsqueeze(-1)
    lrend, rrend = lrend.unsqueeze(-1), rrend.unsqueeze(-1)
    urend, drend = urend.unsqueeze(-1), drend.unsqueeze(-1)

    xc, yc = xc.unsqueeze(-1), yc.unsqueeze(-1)
    w = max(im_size)
    h = min(im_size)
    r = w / h

    xdists = torch.cat(
        ((lobs - xc).abs(), (lrend - xc).abs(),
         (robs - xc).abs(), (rrend - xc).abs()),
        dim=1)
    ydists = torch.cat(
        ((uobs - yc).abs(), (urend - yc).abs(),
         (dobs - yc).abs(), (drend - yc).abs()),
        dim=1)
    xdist = xdists.max(dim=1)[0]
    ydist = ydists.max(dim=1)[0]
    width = torch.max(xdist, ydist * r) * 2 * lamb
    height = torch.max(xdist / r, ydist) * 2 * lamb

    xc, yc = xc.squeeze(-1), yc.squeeze(-1)
    x1, y1, x2, y2 = xc - width/2, yc - height / 2, xc + width / 2, yc + height / 2
    boxes = torch.cat(
        (x1.unsqueeze(1), y1.unsqueeze(1), x2.unsqueeze(1), y2.unsqueeze(1)), dim=1)
    assert not clamp
    if clamp:
        boxes[:, [0, 2]] = torch.clamp(boxes[:, [0, 2]], 0, w - 1)
        boxes[:, [1, 3]] = torch.clamp(boxes[:, [1, 3]], 0, h - 1)
    return boxes

def deepim_crops_robust(images, obs_boxes, K, TCO_pred, O_vertices, output_size=None, lamb=1.4, masks=None, sce_depth=None):
    batch_size, _, h, w = images.shape
    device = images.device
    if output_size is None:
        output_size = (h, w)
    uv = project_points_robust(O_vertices, K, TCO_pred)
    rend_boxes = boxes_from_uv(uv)
    rend_center_uv = project_points_robust(torch.zeros(batch_size, 1, 3).to(device), K, TCO_pred)
    boxes = deepim_boxes(rend_center_uv, obs_boxes, rend_boxes, im_size=(h, w), lamb=lamb)
    boxes = torch.cat((torch.arange(batch_size).unsqueeze(1).to(device).float(), boxes), dim=1)
    crops = torchvision.ops.roi_align(images, boxes, output_size=output_size, sampling_ratio=4)
    masks_crops = None if masks is None else torchvision.ops.roi_align(masks.unsqueeze(1).float(), boxes, output_size=output_size, sampling_ratio=4).squeeze(1).bool()
    sce_depth_crops = None if sce_depth is None else torchvision.ops.roi_align(sce_depth.unsqueeze(1).float(), boxes, output_size=output_size, sampling_ratio=4).squeeze(1)
    return boxes[:, 1:], crops, masks_crops, sce_depth_crops
