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
from sympy import *

cur_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(cur_dir, "../../../../"))
from lib.vis_utils.colormap import colormap

_COLORS = colormap(rgb=True, maximum=1)
color_step = 5

dpi = 300
fig_size = (10, 10)
font_size = 15  # 20
linewidth = 2
marker_size = 5
handlelength = 1.8  # legend label line length
if platform.system() == "Darwin":
    viewer = "open"
else:
    viewer = "eog"


def sigmoid(x):
    return 1 / (1 + exp(-x))


# x_values = np.linspace(-4.5, 3, num=60)  # f
x_values = np.linspace(-7, 7, num=60)  # df
xlim = [x_values.min(), x_values.max()]

x, y = symbols("x y")

# # relu
# f_relu = Piecewise((0, x<0), (x, x>=0))

# silu
f_silu = x * sigmoid(x)
vals_f_silu = [f_silu.subs(x, _xv) for _xv in x_values]

df_silu = diff(f_silu, x)
vals_df_silu = [df_silu.subs(x, _xv) for _xv in x_values]

ddf_silu = diff(df_silu, x)
vals_ddf_silu = [ddf_silu.subs(x, _xv) for _xv in x_values]

# gelu
f_gelu = 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x**3)))
vals_f_gelu = [f_gelu.subs(x, _xv) for _xv in x_values]

df_gelu = diff(f_gelu, x)
vals_df_gelu = [df_gelu.subs(x, _xv) for _xv in x_values]

ddf_gelu = diff(df_gelu, x)
vals_ddf_gelu = [ddf_gelu.subs(x, _xv) for _xv in x_values]

# mish
f_mish = x * tanh(ln(1 + exp(x)))
vals_f_mish = [f_mish.subs(x, _xv) for _xv in x_values]

df_mish = diff(f_mish, x)
vals_df_mish = [df_mish.subs(x, _xv) for _xv in x_values]

ddf_mish = diff(df_mish, x)
vals_ddf_mish = [ddf_mish.subs(x, _xv) for _xv in x_values]


# xlim = [0, 0.062]
# xlim = [-4.5, 3]

# yapf: disable


# yapf: enable

markers = ["o", "s", "^", "x", "v", "*"]
# labels = ["ADD(-S) 0.02d", "ADD(-S) 0.05d", "ADD(-S) 0.1d", "$(2\degree, 2$ cm)", "$2\degree$", "2 cm"]
# labels = ["ADD(-S) 0.02d", "ADD(-S) 0.05d", "ADD(-S) 0.1d", "$2\degree, 2$ cm", "$2\degree$", "2 cm"]


def main_f():
    fig = plt.figure(figsize=(5, 5), dpi=200)
    ########
    # plt.subplot(1, 1, 1)
    plt.grid(True)
    plot_i = 0
    (h1,) = plt.plot(
        x_values,
        vals_f_silu,
        # "--",
        marker=markers[plot_i],
        markersize=marker_size,
        markerfacecolor="none",
        # label=labels[plot_i],
        linewidth=linewidth,
        color=_COLORS[plot_i * color_step],
        # color=(138 / 255.0, 150 / 255.0, 250 / 255.0),
        clip_on=False,
    )

    plot_i += 1
    (h2,) = plt.plot(
        x_values,
        vals_f_gelu,
        # "--",
        marker=markers[plot_i],
        markersize=marker_size,
        markerfacecolor="none",
        # label=labels[plot_i],
        linewidth=linewidth,
        color=_COLORS[plot_i * color_step],
        # color=(138 / 255.0, 150 / 255.0, 250 / 255.0),
        clip_on=False,
    )

    plot_i += 1
    (h3,) = plt.plot(
        x_values,
        vals_f_mish,
        # "--",
        marker=markers[plot_i],
        markersize=marker_size,
        markerfacecolor="none",
        # label=labels[plot_i],
        linewidth=linewidth,
        color=_COLORS[plot_i * color_step],
        # color=(138 / 255.0, 150 / 255.0, 250 / 255.0),
        clip_on=False,
    )

    handles = [h1, h2, h3]
    labels = ["SiLU", "GELU", "Mish"]
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
    # plt.ylim([64, 74])
    # plt.yscale("log")

    ax = plt.gca()
    # ax.xaxis.set_major_locator(MultipleLocator(10))
    # ax.set_xlabel("activations", fontsize=font_size)
    # ax.set_ylabel("accuracy (%)", fontsize=font_size)

    # plt.xticks(region_ids, labels=[str(_r) for _r in regions])
    # ax.set_yticks([60, 70, 80, 90, 100])
    # ax.get_yaxis().set_major_formatter(ticker.ScalarFormatter())

    # ax.xaxis.set_tick_params(labelsize=font_size)
    # ax.yaxis.set_tick_params(labelsize=font_size)
    save_path = "output/activations/activations.pdf"
    mmcv.mkdir_or_exist(osp.dirname(save_path))
    plt.tight_layout()
    plt.savefig(save_path, dpi=fig.dpi)  # , bbox_inches="tight")
    print("save fig path: ", save_path)
    os.system(f"{viewer} {save_path}")


def main_df():
    fig = plt.figure(figsize=(5, 5), dpi=200)
    ########
    # plt.subplot(1, 1, 1)
    plt.grid(True)
    plot_i = 0
    (h1,) = plt.plot(
        x_values,
        vals_df_silu,
        # "--",
        marker=markers[plot_i],
        markersize=marker_size,
        markerfacecolor="none",
        # label=labels[plot_i],
        linewidth=linewidth,
        color=_COLORS[plot_i * color_step],
        # color=(138 / 255.0, 150 / 255.0, 250 / 255.0),
        clip_on=False,
    )

    plot_i += 1
    (h2,) = plt.plot(
        x_values,
        vals_df_gelu,
        # "--",
        marker=markers[plot_i],
        markersize=marker_size,
        markerfacecolor="none",
        # label=labels[plot_i],
        linewidth=linewidth,
        color=_COLORS[plot_i * color_step],
        # color=(138 / 255.0, 150 / 255.0, 250 / 255.0),
        clip_on=False,
    )

    plot_i += 1
    (h3,) = plt.plot(
        x_values,
        vals_df_mish,
        # "--",
        marker=markers[plot_i],
        markersize=marker_size,
        markerfacecolor="none",
        # label=labels[plot_i],
        linewidth=linewidth,
        color=_COLORS[plot_i * color_step],
        # color=(138 / 255.0, 150 / 255.0, 250 / 255.0),
        clip_on=False,
    )

    handles = [h1, h2, h3]
    labels = ["SiLU", "GELU", "Mish"]
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
    # plt.ylim([64, 74])
    # plt.yscale("log")

    ax = plt.gca()
    # ax.xaxis.set_major_locator(MultipleLocator(10))
    # ax.set_xlabel("activations", fontsize=font_size)
    # ax.set_ylabel("accuracy (%)", fontsize=font_size)

    # plt.xticks(region_ids, labels=[str(_r) for _r in regions])
    # ax.set_yticks([60, 70, 80, 90, 100])
    # ax.get_yaxis().set_major_formatter(ticker.ScalarFormatter())

    # ax.xaxis.set_tick_params(labelsize=font_size)
    # ax.yaxis.set_tick_params(labelsize=font_size)
    save_path = "output/activations/activations_df.pdf"
    mmcv.mkdir_or_exist(osp.dirname(save_path))
    plt.tight_layout()
    plt.savefig(save_path, dpi=fig.dpi)  # , bbox_inches="tight")
    print("save fig path: ", save_path)
    os.system(f"{viewer} {save_path}")


def main_ddf():
    fig = plt.figure(figsize=(5, 5), dpi=200)
    ########
    # plt.subplot(1, 1, 1)
    plt.grid(True)
    plot_i = 0
    (h1,) = plt.plot(
        x_values,
        vals_ddf_silu,
        # "--",
        marker=markers[plot_i],
        markersize=marker_size,
        markerfacecolor="none",
        # label=labels[plot_i],
        linewidth=linewidth,
        color=_COLORS[plot_i * color_step],
        # color=(138 / 255.0, 150 / 255.0, 250 / 255.0),
        clip_on=False,
    )

    plot_i += 1
    (h2,) = plt.plot(
        x_values,
        vals_ddf_gelu,
        # "--",
        marker=markers[plot_i],
        markersize=marker_size,
        markerfacecolor="none",
        # label=labels[plot_i],
        linewidth=linewidth,
        color=_COLORS[plot_i * color_step],
        # color=(138 / 255.0, 150 / 255.0, 250 / 255.0),
        clip_on=False,
    )

    plot_i += 1
    (h3,) = plt.plot(
        x_values,
        vals_ddf_mish,
        # "--",
        marker=markers[plot_i],
        markersize=marker_size,
        markerfacecolor="none",
        # label=labels[plot_i],
        linewidth=linewidth,
        color=_COLORS[plot_i * color_step],
        # color=(138 / 255.0, 150 / 255.0, 250 / 255.0),
        clip_on=False,
    )

    handles = [h1, h2, h3]
    labels = ["SiLU", "GELU", "Mish"]
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

    ax = plt.gca()

    save_path = "output/activations/activations_ddf.pdf"
    mmcv.mkdir_or_exist(osp.dirname(save_path))
    plt.tight_layout()
    plt.savefig(save_path, dpi=fig.dpi)  # , bbox_inches="tight")
    print("save fig path: ", save_path)
    os.system(f"{viewer} {save_path}")


if __name__ == "__main__":
    # main_f()
    # main_df()
    main_ddf()
