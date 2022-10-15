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

# yapf: disable
noise_sigmas              = [
    0     , 0.002 , 0.004 , 0.006 , 0.008 , 0.01  , 0.012 , 0.014 , 0.016 , 0.018 , \
    0.02  , 0.022 , 0.024 , 0.026 , 0.028 , 0.03  , 0.032 , 0.034 , 0.036 , 0.038 , \
    0.04  , 0.042 , 0.044 , 0.046 , 0.048 , 0.05  , 0.052 , 0.054 , 0.056 , 0.058 , 0.06  ]
cpnp_add_rel_errors_outlier_10 = [
    0.0160, 0.0157, 0.0157, 0.0163, 0.0164, 0.0160, 0.0162, 0.0161, 0.0158, 0.0161, \
    0.0164, 0.0162, 0.0161, 0.0163, 0.0164, 0.0163, 0.0165, 0.0164, 0.0166, 0.0168, \
    0.0169, 0.0174, 0.0171, 0.0174, 0.0174, 0.0178, 0.0182, 0.0183, 0.0187, 0.0189, 0.0193]
ransanc_pnp_add_rel_errors_outlier_10 = [
    0.0128, 0.0126, 0.0118, 0.0115, 0.0141, 0.0200, 0.0274, 0.0356, 0.0454, 0.0561, \
    0.0672, 0.0819, 0.0946, 0.1036, 0.1162, 0.1300, 0.1443, 0.1590, 0.1805, 0.1933, \
    0.2081, 0.2264, 0.2428, 0.2645, 0.2765, 0.2981, 0.3221, 0.3363, 0.3566, 0.3692, 0.3955]
########
cpnp_add_rel_errors_outlier_30 = [
    0.0269, 0.0264, 0.0265, 0.0262, 0.0261, 0.0260, 0.0261, 0.0255, 0.0262, 0.0260, \
    0.0269, 0.0260, 0.0268, 0.0266, 0.0266, 0.0264, 0.0269, 0.0269, 0.0267, 0.0268, \
    0.0271, 0.0271, 0.0278, 0.0275, 0.0282, 0.0277, 0.0279, 0.0282, 0.0287, 0.0287, 0.0291]
ransanc_pnp_add_rel_errors_outlier_30 = [
    0.0129, 0.0126, 0.0119, 0.0116, 0.0146, 0.0215, 0.0309, 0.0457, 0.0604, 0.0744, \
    0.0891, 0.1027, 0.1192, 0.1341, 0.1471, 0.1620, 0.1787, 0.1961, 0.2047, 0.2294, \
    0.2498, 0.2546, 0.2804, 0.2933, 0.3156, 0.3313, 0.3436, 0.366, 0.3811, 0.431, 0.4455]
# yapf: enable


def log_10_product(x, pos):
    """The two args are the value and tick position.

    Label ticks with the product of the exponentiation
    """
    return "%1i" % x


def main_10():
    print("outlier: 10%")
    fig = plt.figure(figsize=(5, 5), dpi=150)
    plot_i = 0
    (h1,) = plt.plot(
        noise_sigmas,
        ransanc_pnp_add_rel_errors_outlier_10,
        # "-",
        marker="o",
        markersize=marker_size,
        markerfacecolor="none",
        label="RANSAC EPnP",
        linewidth=linewidth,
        color=(255 / 255.0, 150 / 255.0, 150 / 255.0),
    )

    plot_i += 5
    (h2,) = plt.plot(
        noise_sigmas,
        cpnp_add_rel_errors_outlier_10,
        # "-",
        marker="d",
        markersize=marker_size,
        markerfacecolor="none",
        label="Ours",
        linewidth=linewidth,
        color=(0, 112 / 255.0, 68 / 255.0),
    )
    handles = [h1, h2]
    labels = ["RANSAC EPnP", "Ours"]
    plt.legend(
        handles,
        labels,
        loc="upper left",
        # bbox_to_anchor=(0.85, 0.0),
        fontsize=font_size,
        fancybox=True,
        framealpha=0.5,
        handlelength=handlelength,
    )
    plt.xlim([0, 0.062])
    plt.ylim([0.01, 0.45])
    plt.yscale("log")
    plt.grid(True)

    ax = plt.gca()
    # ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.set_xlabel("noise level $\sigma$ (outlier=10%)", fontsize=font_size)
    ax.set_ylabel("pose error", fontsize=font_size)

    ax.set_yticks([0.01, 0.02, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4])
    ax.get_yaxis().set_major_formatter(ticker.ScalarFormatter())

    ax.xaxis.set_tick_params(labelsize=font_size)
    ax.yaxis.set_tick_params(labelsize=font_size)
    save_path = "output/sphere_synt_plot/sphere_outlier_10.png"
    mmcv.mkdir_or_exist(osp.dirname(save_path))
    plt.savefig(save_path, dpi=fig.dpi, bbox_inches="tight")
    print("save fig path: ", save_path)
    os.system(f"{viewer} {save_path}")


def main_30():
    print("outlier: 30%")
    fig = plt.figure(figsize=(5, 5), dpi=150)
    plot_i = 0
    (h1,) = plt.plot(
        noise_sigmas,
        ransanc_pnp_add_rel_errors_outlier_30,
        # "--",
        marker="o",
        markersize=marker_size,
        markerfacecolor="none",
        label="RANSAC EPnP",
        linewidth=linewidth,
        color=(255 / 255.0, 150 / 255.0, 150 / 255.0),
    )
    plot_i += 5

    (h2,) = plt.plot(
        noise_sigmas,
        cpnp_add_rel_errors_outlier_30,
        # "--",
        marker="d",
        markersize=marker_size,
        markerfacecolor="none",
        label="Ours",
        linewidth=linewidth,
        color=(0, 112 / 255.0, 68 / 255.0),
    )
    handles = [h1, h2]
    labels = ["RANSAC EPnP", "Ours"]
    plt.legend(
        handles,
        labels,
        loc="upper left",
        # bbox_to_anchor=(0.85, 0.0),
        fontsize=font_size,
        fancybox=True,
        framealpha=0.5,
        handlelength=handlelength,
    )
    plt.xlim([0, 0.062])
    plt.ylim([0.01, 0.45])
    plt.yscale("log")
    plt.grid(True)

    ax = plt.gca()
    # ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.set_xlabel("noise level $\sigma$ (outlier=30%)", fontsize=font_size)
    ax.set_ylabel("pose error", fontsize=font_size)

    ax.set_yticks([0.01, 0.02, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4])
    ax.get_yaxis().set_major_formatter(ticker.ScalarFormatter())

    ax.xaxis.set_tick_params(labelsize=font_size)
    ax.yaxis.set_tick_params(labelsize=font_size)
    save_path = "output/sphere_synt_plot/sphere_outlier_30.png"
    mmcv.mkdir_or_exist(osp.dirname(save_path))
    plt.savefig(save_path, dpi=fig.dpi, bbox_inches="tight")
    print("save fig path: ", save_path)
    os.system(f"{viewer} {save_path}")


if __name__ == "__main__":
    main_10()
    main_30()
