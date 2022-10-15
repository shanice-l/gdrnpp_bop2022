# https://github.com/DCurro/CannyEdgePytorch/blob/master/canny.py
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import gaussian


class Canny(nn.Module):
    def __init__(self, threshold=10.0, device="cuda"):
        super(Canny, self).__init__()

        self.threshold = threshold
        self.device = torch.device(device)

        filter_size = 5
        generated_filters = gaussian(filter_size, std=1.0).reshape([1, filter_size])

        self.gaussian_filter_horizontal = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(1, filter_size),
            padding=(0, filter_size // 2),
        )
        self.gaussian_filter_horizontal.weight.data.copy_(torch.from_numpy(generated_filters))
        self.gaussian_filter_horizontal.bias.data.copy_(torch.from_numpy(np.array([0.0])))

        self.gaussian_filter_vertical = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(filter_size, 1),
            padding=(filter_size // 2, 0),
        )
        self.gaussian_filter_vertical.weight.data.copy_(torch.from_numpy(generated_filters.T))
        self.gaussian_filter_vertical.bias.data.copy_(torch.from_numpy(np.array([0.0])))

        sobel_filter = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])

        self.sobel_filter_horizontal = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=sobel_filter.shape,
            padding=sobel_filter.shape[0] // 2,
        )
        self.sobel_filter_horizontal.weight.data.copy_(torch.from_numpy(sobel_filter))
        self.sobel_filter_horizontal.bias.data.copy_(torch.from_numpy(np.array([0.0])))

        self.sobel_filter_vertical = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=sobel_filter.shape,
            padding=sobel_filter.shape[0] // 2,
        )
        self.sobel_filter_vertical.weight.data.copy_(torch.from_numpy(sobel_filter.T))
        self.sobel_filter_vertical.bias.data.copy_(torch.from_numpy(np.array([0.0])))

        # filters were flipped manually
        filter_0 = np.array([[0, 0, 0], [0, 1, -1], [0, 0, 0]])

        filter_45 = np.array([[0, 0, 0], [0, 1, 0], [0, 0, -1]])

        filter_90 = np.array([[0, 0, 0], [0, 1, 0], [0, -1, 0]])

        filter_135 = np.array([[0, 0, 0], [0, 1, 0], [-1, 0, 0]])

        filter_180 = np.array([[0, 0, 0], [-1, 1, 0], [0, 0, 0]])

        filter_225 = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 0]])

        filter_270 = np.array([[0, -1, 0], [0, 1, 0], [0, 0, 0]])

        filter_315 = np.array([[0, 0, -1], [0, 1, 0], [0, 0, 0]])

        all_filters = np.stack(
            [
                filter_0,
                filter_45,
                filter_90,
                filter_135,
                filter_180,
                filter_225,
                filter_270,
                filter_315,
            ]
        )

        self.directional_filter = nn.Conv2d(
            in_channels=1,
            out_channels=8,
            kernel_size=filter_0.shape,
            padding=filter_0.shape[-1] // 2,
        )
        self.directional_filter.weight.data.copy_(torch.from_numpy(all_filters[:, None, ...]))
        self.directional_filter.bias.data.copy_(torch.from_numpy(np.zeros(shape=(all_filters.shape[0],))))

    def forward(self, img):
        img_r = img[:, 0:1]
        img_g = img[:, 1:2]
        img_b = img[:, 2:3]

        blur_horizontal = self.gaussian_filter_horizontal(img_r)
        blurred_img_r = self.gaussian_filter_vertical(blur_horizontal)
        blur_horizontal = self.gaussian_filter_horizontal(img_g)
        blurred_img_g = self.gaussian_filter_vertical(blur_horizontal)
        blur_horizontal = self.gaussian_filter_horizontal(img_b)
        blurred_img_b = self.gaussian_filter_vertical(blur_horizontal)

        blurred_img = torch.stack([blurred_img_r, blurred_img_g, blurred_img_b], dim=1)
        blurred_img = torch.stack([torch.squeeze(blurred_img)])

        grad_x_r = self.sobel_filter_horizontal(blurred_img_r)
        grad_y_r = self.sobel_filter_vertical(blurred_img_r)
        grad_x_g = self.sobel_filter_horizontal(blurred_img_g)
        grad_y_g = self.sobel_filter_vertical(blurred_img_g)
        grad_x_b = self.sobel_filter_horizontal(blurred_img_b)
        grad_y_b = self.sobel_filter_vertical(blurred_img_b)

        # COMPUTE THICK EDGES
        grad_mag = torch.sqrt(grad_x_r**2 + grad_y_r**2)
        grad_mag += torch.sqrt(grad_x_g**2 + grad_y_g**2)
        grad_mag += torch.sqrt(grad_x_b**2 + grad_y_b**2)
        grad_orientation = torch.atan2(grad_y_r + grad_y_g + grad_y_b, grad_x_r + grad_x_g + grad_x_b) * (
            180.0 / 3.14159
        )
        grad_orientation += 180.0
        grad_orientation = torch.round(grad_orientation / 45.0) * 45.0

        # THIN EDGES (NON-MAX SUPPRESSION)
        all_filtered = self.directional_filter(grad_mag)

        inidices_positive = (grad_orientation / 45) % 8  # [B, 1, H, W]
        inidices_negative = ((grad_orientation / 45) + 4) % 8

        bs = inidices_positive.size()[0]
        height = inidices_positive.size()[2]
        width = inidices_positive.size()[3]

        pixel_count = height * width
        pixel_range = torch.FloatTensor([range(pixel_count)]).to(device=self.device).repeat(bs, 1).view(-1)

        indices = (inidices_positive.view(-1).data * pixel_count + pixel_range).squeeze()
        channel_select_filtered_positive = all_filtered.view(-1)[indices.long()].view(bs, 1, height, width)

        indices = (inidices_negative.view(-1).data * pixel_count + pixel_range).squeeze()
        channel_select_filtered_negative = all_filtered.view(-1)[indices.long()].view(bs, 1, height, width)

        channel_select_filtered = torch.cat(
            [
                channel_select_filtered_positive,
                channel_select_filtered_negative,
            ],
            dim=1,
        )

        is_max = channel_select_filtered.min(dim=1)[0] > 0.0
        is_max = torch.unsqueeze(is_max, dim=1)

        thin_edges = grad_mag.clone()
        thin_edges[is_max == 0] = 0.0

        # THRESHOLD
        thresholded = thin_edges.clone()
        thresholded[thin_edges < self.threshold] = 0.0

        early_threshold = grad_mag.clone()
        early_threshold[grad_mag < self.threshold] = 0.0

        assert (
            grad_mag.size()
            == grad_orientation.size()
            == thin_edges.size()
            == thresholded.size()
            == early_threshold.size()
        )

        return (
            blurred_img,
            grad_mag,
            grad_orientation,
            thin_edges,
            thresholded,
            early_threshold,
        )


def get_canny_edge_raw(raw_img, threshold=3.0, device="cuda"):
    """rgb image: HWC."""
    assert raw_img.shape[-1] == 3, raw_img.shape
    img = torch.from_numpy(raw_img.transpose((2, 0, 1)))  # CHW
    batch = torch.stack([img]).float()  # 1CHW

    net = Canny(threshold=threshold, device=device).to(device)
    net.eval()

    data = batch.to(device)

    with torch.no_grad():
        (
            blurred_img,
            grad_mag,
            grad_orientation,
            thin_edges,
            thresholded,
            early_threshold,
        ) = net(data)
    return thresholded
    # imsave('gradient_magnitude.png',grad_mag.data.cpu().numpy()[0,0])
    # imsave('thin_edges.png', thresholded.data.cpu().numpy()[0, 0])
    # imsave('final.png', (thresholded.data.cpu().numpy()[0, 0] > 0.0).astype(float))
    # imsave('thresholded.png', early_threshold.data.cpu().numpy()[0, 0])


def get_canny_edge_tensor(img_tensor, threshold=3.0, device="cuda"):
    """NCHW, rgb."""
    assert img_tensor.shape[1] == 3, img_tensor.shape
    net = Canny(threshold=threshold, device=device).to(device)
    net.eval()

    data = img_tensor.to(device)
    with torch.no_grad():
        (
            blurred_img,
            grad_mag,
            grad_orientation,
            thin_edges,
            thresholded,
            early_threshold,
        ) = net(data)
    return thresholded


def mask_dilate_th(mask, kernel_size=3):
    """
    Args:
        mask: [B,1,H,W]
    """
    if isinstance(kernel_size, (int, float)):
        kernel_size = (int(kernel_size), int(kernel_size))
    padding = (kernel_size[0] // 2, kernel_size[1] // 2)
    kernel = torch.ones(kernel_size)[None, None].to(mask)
    result = torch.clamp(F.conv2d(mask, kernel, padding=padding), 0, 1)
    return result


def mask_erode_th(mask, kernel_size=3):
    """
    Args:
        mask: [B,1,H,W]
    """
    if isinstance(kernel_size, (int, float)):
        kernel_size = (int(kernel_size), int(kernel_size))
    padding = (kernel_size[0] // 2, kernel_size[1] // 2)
    kernel = torch.ones(kernel_size)[None, None].to(mask)

    # dilate bg mask
    result = torch.clamp(F.conv2d(1 - mask, kernel, padding=padding), 0, 1)
    # invert
    return (1 - result).to(torch.float32)


def compute_mask_edge_weights(
    mask,
    dilate_kernel_size=5,
    erode_kernel_size=5,
    w_edge=5.0,
    edge_lower=True,
):
    """defined in Contour Loss: Boundary-Aware Learning for Salient Object Segmentation
    (https://arxiv.org/abs/1908.01975)
    Args:
        mask: [B, 1, H, W]
    """
    dilated_mask = mask_dilate_th(mask, kernel_size=dilate_kernel_size)
    eroded_mask = mask_erode_th(mask, kernel_size=erode_kernel_size)
    # edge width: kd//2 + ke//2 ?
    mask_edge = dilated_mask - eroded_mask
    # old (bug, edge has lower weight)
    if edge_lower:
        # >1 for non-edge, ~1 for edge
        return torch.exp(-0.5 * (mask_edge * w_edge) ** 2) / (math.sqrt(2 * math.pi)) + 1
    else:
        # 1 for non-edge, >1 for edge
        _gaussian = torch.exp(-0.5 * (mask_edge * w_edge) ** 2) / (math.sqrt(2 * math.pi))
        return _gaussian.max() - _gaussian + 1  # new


def get_edge_weights_canny_mask(
    image,
    mask,
    dilate_kernel_size=11,
    erode_kernel_size=11,
    w_edge=1.0,
    canny_thr=3.0,
):
    """
    Args:
        image: NCHW
        mask: N1HW
    """
    edge_map = get_canny_edge_tensor(image, canny_thr, device=image.device)
    mask_edge_weight = compute_mask_edge_weights(mask, dilate_kernel_size, erode_kernel_size, w_edge)
    final_weight = (edge_map + mask_edge_weight).sigmoid()
    return final_weight


# ==========================================
def test_edge_weight():
    device = "cuda"
    data = mmcv.load("tmp_data.pkl")
    gt_images = data["gt_images"]
    pred_masks = data["pred_masks"]

    # show_idx = random.randint(0, len(gt_images)-1)

    # eroded_mask = cv2.erode(fg_mask.astype(np.uint8),
    #         kernel=np.ones((8, 8), dtype=np.uint8)).astype(np.float32)
    mask_tensor = torch.as_tensor(pred_masks, dtype=torch.float32).to(device=device)
    img_tensor = torch.as_tensor(gt_images).to(device=device)  # NCHW, BGR
    edge_weight = compute_mask_edge_weights(
        mask_tensor,
        dilate_kernel_size=11,
        erode_kernel_size=11,
        edge_lower=False,
    )
    # edge_weight = get_edge_weights_canny_mask(img_tensor, mask_tensor)
    print(edge_weight.shape)

    for show_idx in range(0, len(gt_images)):
        imgcolor = gt_images[show_idx][[2, 1, 0], :, :].transpose(1, 2, 0)  # HWC, rgb
        fg_mask = pred_masks[show_idx, 0]
        edge_weight_show = edge_weight[show_idx, 0].detach().cpu().numpy()
        print(edge_weight_show.max(), edge_weight_show.min())
        edge_weight_show = edge_weight_show / edge_weight_show.max()
        grid_show(
            [imgcolor, fg_mask, edge_weight_show],
            ["bgr", "fg_mask", "edge_weight"],
            row=2,
            col=2,
        )


def test_canny():
    data = mmcv.load("tmp_data.pkl")
    gt_images = data["gt_images"]
    pred_masks = data["pred_masks"]
    ren_images = data["ren_images"]

    imgcolor = gt_images[0][[2, 1, 0], :, :].transpose(1, 2, 0)  # HWC, rgb

    fg_mask = pred_masks[0, 0]
    # eroded_mask = cv2.erode(fg_mask.astype(np.uint8),
    #         kernel=np.ones((8, 8), dtype=np.uint8)).astype(np.float32)
    eroded_mask = (
        mask_erode_th(
            torch.as_tensor(fg_mask, dtype=torch.float32)[None, None],
            kernel_size=11,
        )
        .detach()
        .cpu()
        .numpy()
    )

    dilated_mask = (
        mask_dilate_th(
            torch.as_tensor(fg_mask, dtype=torch.float32)[None, None],
            kernel_size=11,
        )
        .detach()
        .cpu()
        .numpy()
    )
    print(dilated_mask.shape)
    edge_map = get_canny_edge_raw(imgcolor).detach().cpu().numpy() * (dilated_mask * (1 - eroded_mask))
    edge_map[edge_map > 0] = 1

    edge_map_naive = get_canny_edge_raw(imgcolor).detach().cpu().numpy()
    edge_map_naive[edge_map_naive > 0] = 1
    edge_map_naive[0, 0]

    ren_image = ren_images[0].transpose(1, 2, 0)  # HWC, bgr
    cv2.imwrite("tmp_gt.png", (imgcolor * 255 + 0.5).astype("uint8"))
    cv2.imwrite("tmp_ren.png", (ren_image * 255 + 0.5).astype("uint8"))
    grid_show(
        [
            imgcolor,
            ren_image[:, :, [2, 1, 0]],
            fg_mask,
            edge_map[0, 0],
            edge_map_naive[0, 0],
        ],
        ["bgr", "ren_image", "fg_mask", "edge", "edge_naive"],
        row=2,
        col=3,
    )


if __name__ == "__main__":
    import cv2
    import random
    import mmcv
    from lib.vis_utils.image import grid_show
    import os.path as osp
    import sys
    import kornia

    cur_dir = osp.dirname(__file__)
    sys.path.append(osp.join(cur_dir, "../.."))

    test_edge_weight()
    # test_canny()

    # im_path = 'datasets/BOP_DATASETS/lm/test/000002/rgb/000000.png'
    # assert osp.exists(im_path), im_path
    # img_np = cv2.imread(im_path, cv2.IMREAD_COLOR)[:, :, [2,1,0]] / 255.0

    # img = torch.as_tensor(img_np, dtype=torch.float32)[None].permute(0, 3, 1, 2)

    # from core.utils import rgb_lab
    # img_lab = rgb_lab.rgb_to_lab(img.permute(0, 2,3,1))
    # img_lab_normed = rgb_lab.normalize_lab(img_lab)
    # thresholded = get_canny_edge_tensor(img_lab_normed.permute(0,3,1,2), device='cuda', threshold=3.0)

    # # thresholded = get_canny_edge_tensor(img, device='cuda', threshold=3.0)

    # # thresholded = get_canny_edge_tensor(kornia.rgb_to_grayscale(img).repeat(1, 3, 1, 1), device='cuda', threshold=3.0)
    # thresholded = (thresholded > 0).float()
    # edge = thresholded.detach().cpu().numpy()[0,0]
    # print(edge.min(), edge.max(), edge.mean())
    # cv2.imshow('rgb', img_np[:,:,[2,1,0]])
    # cv2.imshow("edge", edge)
    # k = cv2.waitKey()
    # if k == 27:
    #     cv2.destroyAllWindows()
