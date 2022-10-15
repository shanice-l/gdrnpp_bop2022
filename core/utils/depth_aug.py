import numpy as np
import random


def add_noise_depth(depth, level=0.005, depth_valid_min=0):
    # from DeepIM-PyTorch and se3tracknet
    # in deepim: level=0.1, valid_min=0
    # in se3tracknet, level=5/1000, depth_valid_min = 100/1000 = 0.1

    if len(depth.shape) == 3:
        mask = depth[:, :, -1] > depth_valid_min
        row, col, ch = depth.shape
        noise_level = random.uniform(0, level)
        gauss = noise_level * np.random.randn(row, col)
        gauss = np.repeat(gauss[:, :, np.newaxis], ch, axis=2)
    else:  # 2
        mask = depth > depth_valid_min
        row, col = depth.shape
        noise_level = random.uniform(0, level)
        gauss = noise_level * np.random.randn(row, col)
    noisy = depth.copy()
    noisy[mask] = depth[mask] + gauss[mask]
    return noisy


if __name__ == "__main__":
    from lib.vis_utils.image import heatmap, grid_show
    import mmcv
    import cv2
    from skimage.restoration import denoise_bilateral

    # depth = mmcv.imread("datasets/BOP_DATASETS/ycbv/train_pbr/000000/depth/000000.png", "unchanged") / 10000.0
    # depth_aug = add_noise_depth(depth, level=0.005, depth_valid_min=0.1)
    # diff = depth_aug - depth
    # grid_show([
    #     heatmap(depth, to_rgb=True), heatmap(depth_aug, to_rgb=True),
    #     heatmap(diff, to_rgb=True)
    # ], ["depth", "depth_aug", "diff"], row=1, col=3)

    depth = (mmcv.imread("datasets/BOP_DATASETS/ycbv/test/000048/depth/000001.png", "unchanged") / 10000.0).astype(
        "float32"
    )
    # diameter, pix_sigma, space_sigma
    depth_aug = cv2.bilateralFilter(depth, 11, 0.1, 30)
    # depth_aug = denoise_bilateral(depth, sigma_color=0.05, sigma_spatial=15)
    diff = depth_aug - depth
    grid_show(
        [heatmap(depth, to_rgb=True), heatmap(depth_aug, to_rgb=True), heatmap(diff, to_rgb=True)],
        ["depth", "depth_aug", "diff"],
        row=1,
        col=3,
    )
