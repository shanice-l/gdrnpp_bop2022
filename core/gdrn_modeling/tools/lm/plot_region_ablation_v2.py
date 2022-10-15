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
import matplotlib
from tempfile import NamedTemporaryFile

cur_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(cur_dir, "../../../../"))
from lib.vis_utils.colormap import colormap

_COLORS = colormap(rgb=True, maximum=1)
color_step = 8

dpi = 300
fig_size = (10, 10)
font_size = 15  # 20
linewidth = 2
marker_size = 8
handlelength = 1.8  # legend label line length
if platform.system() == "Darwin":
    viewer = "open"
else:
    viewer = "eog"


def get_size(fig, dpi=100):
    with NamedTemporaryFile(suffix=".png") as f:
        fig.savefig(f.name, bbox_inches="tight", dpi=dpi)
        height, width, _channels = matplotlib.image.imread(f.name).shape
        return width / dpi, height / dpi


def set_size(fig, size, dpi=100, eps=1e-2, give_up=2, min_size_px=10):
    target_width, target_height = size
    set_width, set_height = (
        target_width,
        target_height,
    )  # reasonable starting point
    deltas = []  # how far we have
    while True:
        fig.set_size_inches([set_width, set_height])
        actual_width, actual_height = get_size(fig, dpi=dpi)
        set_width *= target_width / actual_width
        set_height *= target_height / actual_height
        deltas.append(abs(actual_width - target_width) + abs(actual_height - target_height))
        if deltas[-1] < eps:
            return True
        if len(deltas) > give_up and sorted(deltas[-give_up:]) == deltas[-give_up:]:
            return False
        if set_width * dpi < min_size_px or set_height * dpi < min_size_px:
            return False


# xlim = [0, 0.062]
xlim = [0, 8]

# yapf: disable
region_ids = [
    0    , 1    , 2    , 3    , 4    , 5    , 6    , 7    ,   8]
regions = [
    0    , 1    , 4    , 8    , 16   , 32   , 64   , 128  , 256]

ad_2_list = [
    33.67, 34.63, 34.46, 34.79, 35.27, 35.00, 35.51, 35.69, 33.92]
ad_5_list = [
    74.3, 75.4, 75.5, 74.8, 75.2, 74.9, 76.3, 76.0, 74.0]
ad_10_list = [
    92.98, 93.51, 93.33, 93.36, 93.61, 93.26, 93.69, 93.54, 92.14]
rete_2_list = [
    60.46, 58.91, 59.46, 60.80, 60.97, 62.05, 62.11, 62.12, 62.70]
re_2_list = [
    61.84, 60.15, 60.49, 62.00, 62.13, 63.43, 63.18, 63.41, 64.31]
te_2_list = [
    94.94, 95.22, 95.10, 95.24, 95.39, 95.10, 95.48, 95.56, 95.14]
mean_list = [
    69.7, 69.6, 69.7, 70.2, 70.4, 70.6, 71.0, 71.0, 70.4]
# yapf: enable

markers = ["o", "s", "*", "x", "v", "^"]
labels = [
    "ADD(-S) 0.02d",
    "ADD(-S) 0.05d",
    "ADD(-S) 0.1d",
    "$(2\degree, 2$ cm)",
    "$2\degree$",
    "2 cm",
]


def main():
    print("ablation for regions")
    fig = plt.figure(figsize=(5, 5), dpi=150)
    # fig = plt.figure()
    plt.grid(True)

    plot_i = 0
    (h1,) = plt.plot(
        region_ids,
        ad_2_list,
        "--",
        # marker="s",
        marker=markers[plot_i],
        markersize=marker_size,
        markerfacecolor="none",
        # label="Single-Stage",
        label=labels[plot_i],
        linewidth=linewidth,
        color=_COLORS[plot_i * color_step],
        # color=(138 / 255.0, 150 / 255.0, 250 / 255.0),
        clip_on=False,
    )

    plot_i += 1
    (h2,) = plt.plot(
        region_ids,
        ad_5_list,
        "--",
        # marker="s",
        marker=markers[plot_i],
        markersize=marker_size,
        markerfacecolor="none",
        # label="Single-Stage",
        label=labels[plot_i],
        linewidth=linewidth,
        color=_COLORS[plot_i * color_step],
        # color=(138 / 255.0, 150 / 255.0, 250 / 255.0),
        clip_on=False,
    )

    plot_i += 1
    (h3,) = plt.plot(
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
        color=_COLORS[plot_i * color_step],
        # color=(138 / 255.0, 150 / 255.0, 250 / 255.0),
        clip_on=False,
    )

    plot_i += 1
    (h4,) = plt.plot(
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
        color=_COLORS[plot_i * color_step],
        # color=(0, 112 / 255.0, 68 / 255.0),
        clip_on=False,
    )

    plot_i += 1
    (h5,) = plt.plot(
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
        color=_COLORS[plot_i * color_step],
        # color=(0, 112 / 255.0, 68 / 255.0),
        clip_on=False,
    )

    plot_i += 1
    (h6,) = plt.plot(
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
        color=_COLORS[plot_i * color_step],
        # color=(0, 112 / 255.0, 68 / 255.0),
        clip_on=False,
    )

    # handles = [h1, h2, h3, h4, h5]
    # handles = [h1, h2, h3, h4, h5, h6]
    # plt.legend(
    #     handles,
    #     labels,
    #     # loc="upper left",
    #     loc="best",
    #     # bbox_to_anchor=(0.85, 0.0),
    #     # ncol=2,
    #     fontsize=font_size,
    #     fancybox=True,
    #     framealpha=0.5,
    #     handlelength=handlelength,
    # )
    plt.xlim(xlim)
    plt.ylim([30, 100])
    # plt.ylim([50, 100])
    # plt.yscale("log")

    ax = plt.gca()
    # ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.set_xlabel("number of regions\n(a)", fontsize=font_size)
    ax.set_ylabel("accuracy (%)", fontsize=font_size)

    plt.xticks(region_ids, labels=[str(_r) for _r in regions])
    ax.set_yticks([30, 40, 50, 60, 70, 80, 90, 100])
    # ax.get_yaxis().set_major_formatter(ticker.ScalarFormatter())

    ax.xaxis.set_tick_params(labelsize=font_size)
    ax.yaxis.set_tick_params(labelsize=font_size)
    save_path = "output/lm/ablation_regions_v3.png"
    mmcv.mkdir_or_exist(osp.dirname(save_path))
    # set_size(fig, fig_size, dpi=dpi)
    plt.tight_layout()
    plt.savefig(save_path, dpi=fig.dpi)  # , bbox_inches="tight")
    print("save fig path: ", save_path)
    os.system(f"{viewer} {save_path}")


def main_mean():
    fig = plt.figure(figsize=(5, 5), dpi=150)
    # fig = plt.figure()
    plt.grid(True)

    ########
    plot_i = 0
    (h1,) = plt.plot(
        [0],
        [0],
        "--",
        # marker="s",
        marker=markers[plot_i],
        markersize=marker_size,
        markerfacecolor="none",
        # label="Single-Stage",
        label=labels[plot_i],
        linewidth=linewidth,
        color=_COLORS[plot_i * color_step],
        # color=(138 / 255.0, 150 / 255.0, 250 / 255.0),
        clip_on=False,
    )

    plot_i += 1
    (h2,) = plt.plot(
        [0],
        [0],
        "--",
        # marker="s",
        marker=markers[plot_i],
        markersize=marker_size,
        markerfacecolor="none",
        # label="Single-Stage",
        label=labels[plot_i],
        linewidth=linewidth,
        color=_COLORS[plot_i * color_step],
        # color=(138 / 255.0, 150 / 255.0, 250 / 255.0),
        clip_on=False,
    )

    plot_i += 1
    (h3,) = plt.plot(
        [0],
        [0],
        "--",
        # marker="s",
        marker=markers[plot_i],
        markersize=marker_size,
        markerfacecolor="none",
        # label="Single-Stage",
        label=labels[plot_i],
        linewidth=linewidth,
        color=_COLORS[plot_i * color_step],
        # color=(138 / 255.0, 150 / 255.0, 250 / 255.0),
        clip_on=False,
    )

    plot_i += 1
    (h4,) = plt.plot(
        [0],
        [0],
        "--",
        # marker="d",
        marker=markers[plot_i],
        markersize=marker_size,
        markerfacecolor="none",
        # label="Ours",
        label=labels[plot_i],
        linewidth=linewidth,
        color=_COLORS[plot_i * color_step],
        # color=(0, 112 / 255.0, 68 / 255.0),
        clip_on=False,
    )

    plot_i += 1
    (h5,) = plt.plot(
        [0],
        [0],
        "--",
        # marker="d",
        marker=markers[plot_i],
        markersize=marker_size,
        markerfacecolor="none",
        # label="Ours",
        label=labels[plot_i],
        linewidth=linewidth,
        color=_COLORS[plot_i * color_step],
        # color=(0, 112 / 255.0, 68 / 255.0),
        clip_on=False,
    )

    plot_i += 1
    (h6,) = plt.plot(
        [0],
        [0],
        "--",
        # marker="d",
        marker=markers[plot_i],
        markersize=marker_size,
        markerfacecolor="none",
        # label="Ours",
        label=labels[plot_i],
        linewidth=linewidth,
        color=_COLORS[plot_i * color_step],
        # color=(0, 112 / 255.0, 68 / 255.0),
        clip_on=False,
    )
    ########

    # plot_i = 0
    (h7,) = plt.plot(
        region_ids,
        mean_list,
        # "-",
        marker="d",
        # marker=markers[-1],
        markersize=marker_size,
        markerfacecolor="none",
        # label="RANSAC EPnP",
        label=labels[-1],
        linewidth=linewidth,
        # color=_COLORS[plot_i*5],
        color=(0, 112 / 255.0, 68 / 255.0),
        clip_on=False,
    )

    handles = [h1, h2, h3, h4, h5, h6, h7]
    # handles = [h7]
    plt.legend(
        handles,
        labels + ["MEAN"],
        loc="lower right",
        # loc="center left",
        # bbox_to_anchor=(0.85, 0.0),
        fontsize=font_size,
        fancybox=True,
        framealpha=0.5,
        handlelength=handlelength,
    )
    plt.xlim(xlim)
    # plt.ylim([30, 100])
    plt.ylim([64, 74])
    # plt.yscale("log")

    ax = plt.gca()
    # ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.set_xlabel("number of regions\n(b)", fontsize=font_size)
    ax.set_ylabel("accuracy (%)", fontsize=font_size)

    plt.xticks(region_ids, labels=[str(_r) for _r in regions])
    # ax.set_yticks([60, 70, 80, 90, 100])
    # ax.get_yaxis().set_major_formatter(ticker.ScalarFormatter())

    ax.xaxis.set_tick_params(labelsize=font_size)
    ax.yaxis.set_tick_params(labelsize=font_size)
    save_path = "output/lm/ablation_regions_v3_mean.png"
    mmcv.mkdir_or_exist(osp.dirname(save_path))
    # set_size(fig, fig_size, dpi=dpi)
    plt.tight_layout()
    plt.savefig(save_path, dpi=fig.dpi)  # , bbox_inches="tight")
    print("save fig path: ", save_path)
    os.system(f"{viewer} {save_path}")


if __name__ == "__main__":
    main()
    main_mean()
