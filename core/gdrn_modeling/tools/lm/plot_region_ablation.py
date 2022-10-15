import matplotlib.pyplot as plt
from matplotlib import ticker
import os
import os.path as osp
import numpy as np
import sys
import mmcv
from matplotlib.pyplot import axvline
from matplotlib.pyplot import MultipleLocator
import platform

cur_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(cur_dir, "../../../../"))
from lib.vis_utils.colormap import colormap

_COLORS = colormap(rgb=True, maximum=1)

font_size = 15  # 20
linewidth = 2
marker_size = 8
handlelength = 1.8  # legend label line length
if platform.system() == "Darwin":
    viewer = "open"
else:
    viewer = "eog"

# xlim = [0, 0.062]
xlim = [0, 8]

# yapf: disable
region_ids = [
    0    , 1    , 2    , 3    , 4    , 5    , 6    , 7    ,   8]
regions = [
    0    , 1    , 4    , 8    , 16   , 32   , 64   , 128  , 256]

ad_2_list = [
    33.67, 34.63, 34.46, 34.79, 35.27, 35.00, 35.51, 35.69, 33.92]
ad_10_list = [
    92.98, 93.51, 93.33, 93.36, 93.61, 93.26, 93.69, 93.54, 92.14]
rete_2_list = [
    60.46, 58.91, 59.46, 60.80, 60.97, 62.05, 62.11, 62.12, 62.70]
re_2_list = [
    61.84, 60.15, 60.49, 62.00, 62.13, 63.43, 63.18, 63.41, 64.31]
te_2_list = [
    94.94, 95.22, 95.10, 95.24, 95.39, 95.10, 95.48, 95.56, 95.14]
mean_list = [
    77.60, 76.90, 77.10, 77.90, 78.00, 78.50, 78.60, 78.70, 78.60]
# yapf: enable

markers = ["o", "s", "*", "x", "d"]
labels = ["ADD(-S)", "$2\degree, 2$ cm", "$2\degree$", "2 cm", "MEAN"]


def main():
    print("ablation for regions: 10%")
    fig = plt.figure(figsize=(10, 5), dpi=150)
    plt.grid(True)

    plot_i = 0
    (h2,) = plt.plot(
        region_ids,
        ad_10_list,
        "--",
        # marker="s",
        marker=markers[plot_i],
        markersize=marker_size,
        markerfacecolor="none",
        # label="Single-Stage",
        label=labels[plot_i],
        linewidth=linewidth,
        color=_COLORS[plot_i * 5],
        # color=(138 / 255.0, 150 / 255.0, 250 / 255.0),
        clip_on=False,
    )

    plot_i += 1
    (h3,) = plt.plot(
        region_ids,
        rete_2_list,
        "--",
        # marker="d",
        marker=markers[plot_i],
        markersize=marker_size,
        markerfacecolor="none",
        # label="Ours",
        label=labels[plot_i],
        linewidth=linewidth,
        color=_COLORS[plot_i * 5],
        # color=(0, 112 / 255.0, 68 / 255.0),
        clip_on=False,
    )

    plot_i += 1
    (h4,) = plt.plot(
        region_ids,
        re_2_list,
        "--",
        # marker="d",
        marker=markers[plot_i],
        markersize=marker_size,
        markerfacecolor="none",
        # label="Ours",
        label=labels[plot_i],
        linewidth=linewidth,
        color=_COLORS[plot_i * 5],
        # color=(0, 112 / 255.0, 68 / 255.0),
        clip_on=False,
    )

    plot_i += 1
    (h5,) = plt.plot(
        region_ids,
        te_2_list,
        "--",
        # marker="d",
        marker=markers[plot_i],
        markersize=marker_size,
        markerfacecolor="none",
        # label="Ours",
        label=labels[plot_i],
        linewidth=linewidth,
        color=_COLORS[plot_i * 5],
        # color=(0, 112 / 255.0, 68 / 255.0),
        clip_on=False,
    )

    plot_i += 1
    (h1,) = plt.plot(
        region_ids,
        mean_list,
        # "-",
        # marker="o",
        marker=markers[plot_i],
        markersize=marker_size,
        markerfacecolor="none",
        # label="RANSAC EPnP",
        label=labels[plot_i],
        linewidth=linewidth,
        # color=_COLORS[plot_i*5],
        color=(0, 112 / 255.0, 68 / 255.0),
        clip_on=False,
    )

    # handles = [h1, h2, h3, h4, h5]
    handles = [h2, h3, h4, h5, h1]
    plt.legend(
        handles,
        labels,
        loc="upper left",
        # loc="center left",
        # bbox_to_anchor=(0.85, 0.0),
        fontsize=font_size,
        fancybox=True,
        framealpha=0.5,
        handlelength=handlelength,
    )
    plt.xlim(xlim)
    # plt.ylim([30, 100])
    plt.ylim([57, 100])
    # plt.yscale("log")

    ax = plt.gca()
    # ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.set_xlabel("number of regions", fontsize=font_size)
    ax.set_ylabel("accuracy (%)", fontsize=font_size)

    plt.xticks(region_ids, labels=[str(_r) for _r in regions])
    ax.set_yticks([60, 70, 80, 90, 100])
    # ax.get_yaxis().set_major_formatter(ticker.ScalarFormatter())

    ax.xaxis.set_tick_params(labelsize=font_size)
    ax.yaxis.set_tick_params(labelsize=font_size)
    save_path = "output/lm/ablation_regions.png"
    mmcv.mkdir_or_exist(osp.dirname(save_path))
    plt.savefig(save_path, dpi=fig.dpi, bbox_inches="tight")
    print("save fig path: ", save_path)
    os.system(f"{viewer} {save_path}")


if __name__ == "__main__":
    main()
