import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
from mmcv.cnn import normal_init, constant_init
from lib.torch_utils.layers.layer_utils import get_norm, get_nn_act_func
from lib.torch_utils.layers.conv_module import ConvModule
from lib.torch_utils.layers.std_conv_transpose import StdConvTranspose2d


class TopDownDoubleMaskXyzRegionHead(nn.Module):
    def __init__(
        self,
        in_dim,
        up_types=("deconv", "bilinear", "bilinear"),
        deconv_kernel_size=3,
        num_conv_per_block=2,
        feat_dim=256,
        feat_kernel_size=3,
        use_ws=False,
        use_ws_deconv=False,
        norm="GN",
        num_gn_groups=32,
        act="GELU",
        out_kernel_size=1,
        out_layer_shared=True,
        mask_num_classes=1,
        xyz_num_classes=1,
        region_num_classes=1,
        mask_out_dim=2,
        xyz_out_dim=3,
        region_out_dim=65,  # 64+1
    ):
        """
        Args:
            up_types: use up-conv or deconv for each up-sampling layer
                ("bilinear", "bilinear", "bilinear")
                ("deconv", "bilinear", "bilinear")  # CDPNv2 rot head
                ("deconv", "deconv", "deconv")  # CDPNv1 rot head
                ("nearest", "nearest", "nearest")  # implement here but maybe won't use
        NOTE: default from stride 32 to stride 4 (3 ups)
        """
        super().__init__()
        assert out_kernel_size in [
            1,
            3,
        ], "Only support output kernel size: 1 and 3"
        assert deconv_kernel_size in [
            1,
            3,
            4,
        ], "Only support deconv kernel size: 1, 3, and 4"
        assert len(up_types) > 0, up_types

        self.features = nn.ModuleList()
        for i, up_type in enumerate(up_types):
            _in_dim = in_dim if i == 0 else feat_dim
            if up_type == "deconv":
                (
                    deconv_kernel,
                    deconv_pad,
                    deconv_out_pad,
                ) = _get_deconv_pad_outpad(deconv_kernel_size)
                deconv_layer = StdConvTranspose2d if use_ws_deconv else nn.ConvTranspose2d
                self.features.append(
                    deconv_layer(
                        _in_dim,
                        feat_dim,
                        kernel_size=deconv_kernel,
                        stride=2,
                        padding=deconv_pad,
                        output_padding=deconv_out_pad,
                        bias=False,
                    )
                )
                self.features.append(get_norm(norm, feat_dim, num_gn_groups=num_gn_groups))
                self.features.append(get_nn_act_func(act))
            elif up_type == "bilinear":
                self.features.append(nn.UpsamplingBilinear2d(scale_factor=2))
            elif up_type == "nearest":
                self.features.append(nn.UpsamplingNearest2d(scale_factor=2))
            else:
                raise ValueError(f"Unknown up_type: {up_type}")

            if up_type in ["bilinear", "nearest"]:
                assert num_conv_per_block >= 1, num_conv_per_block
            for i_conv in range(num_conv_per_block):
                if i == 0 and i_conv == 0 and up_type in ["bilinear", "nearest"]:
                    conv_in_dim = in_dim
                else:
                    conv_in_dim = feat_dim

                if use_ws:
                    conv_cfg = dict(type="StdConv2d")
                else:
                    conv_cfg = None

                self.features.append(
                    ConvModule(
                        conv_in_dim,
                        feat_dim,
                        kernel_size=feat_kernel_size,
                        padding=(feat_kernel_size - 1) // 2,
                        conv_cfg=conv_cfg,
                        norm=norm,
                        num_gn_groups=num_gn_groups,
                        act=act,
                    )
                )

        self.out_layer_shared = out_layer_shared
        self.mask_num_classes = mask_num_classes
        self.xyz_num_classes = xyz_num_classes
        self.region_num_classes = region_num_classes

        self.mask_out_dim = mask_out_dim
        self.xyz_out_dim = xyz_out_dim
        self.region_out_dim = region_out_dim

        if self.out_layer_shared:
            out_dim = (
                self.mask_out_dim * self.mask_num_classes
                + self.xyz_out_dim * self.xyz_num_classes
                + self.region_out_dim * self.region_num_classes
            )
            self.out_layer = nn.Conv2d(
                feat_dim,
                out_dim,
                kernel_size=out_kernel_size,
                padding=(out_kernel_size - 1) // 2,
                bias=True,
            )
        else:
            self.vis_mask_out_layer = nn.Conv2d(
                feat_dim,
                (self.mask_out_dim // 2) * self.mask_num_classes,
                kernel_size=out_kernel_size,
                padding=(out_kernel_size - 1) // 2,
                bias=True,
            )
            self.full_mask_out_layer = nn.Conv2d(
                feat_dim,
                (self.mask_out_dim // 2) * self.mask_num_classes,
                kernel_size=out_kernel_size,
                padding=(out_kernel_size - 1) // 2,
                bias=True,
            )
            self.xyz_out_layer = nn.Conv2d(
                feat_dim,
                self.xyz_out_dim * self.xyz_num_classes,
                kernel_size=out_kernel_size,
                padding=(out_kernel_size - 1) // 2,
                bias=True,
            )
            self.region_out_layer = nn.Conv2d(
                feat_dim,
                self.region_out_dim * self.region_num_classes,
                kernel_size=out_kernel_size,
                padding=(out_kernel_size - 1) // 2,
                bias=True,
            )

        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)
            elif isinstance(m, nn.ConvTranspose2d):
                normal_init(m, std=0.001)
        # init output layers
        if self.out_layer_shared:
            normal_init(self.out_layer, std=0.01)
        else:
            normal_init(self.mask_out_layer, std=0.01)
            normal_init(self.xyz_out_layer, std=0.01)
            normal_init(self.region_out_layer, std=0.01)

    def forward(self, x):
        if isinstance(x, (tuple, list)) and len(x) == 1:
            x = x[0]
        for i, l in enumerate(self.features):
            x = l(x)
        if self.out_layer_shared:
            out = self.out_layer(x)
            mask_dim = self.mask_out_dim * self.mask_num_classes
            vis_mask = out[:, : (mask_dim // 2), :, :]
            full_mask = out[:, (mask_dim // 2) : mask_dim, :, :]

            xyz_dim = self.xyz_out_dim * self.xyz_num_classes
            xyz = out[:, mask_dim : mask_dim + xyz_dim, :, :]

            region = out[:, mask_dim + xyz_dim :, :, :]

            bs, c, h, w = xyz.shape
            xyz = xyz.view(bs, 3, xyz_dim // 3, h, w)
            coor_x = xyz[:, 0, :, :, :]
            coor_y = xyz[:, 1, :, :, :]
            coor_z = xyz[:, 2, :, :, :]

        else:
            vis_mask = self.vis_mask_out_layer(x)
            full_mask = self.full_mask_out_layer(x)

            xyz = self.xyz_out_layer(x)
            bs, c, h, w = xyz.shape
            xyz = xyz.view(bs, 3, c // 3, h, w)
            coor_x = xyz[:, 0, :, :, :]
            coor_y = xyz[:, 1, :, :, :]
            coor_z = xyz[:, 2, :, :, :]

            region = self.region_out_layer(x)
        return vis_mask, full_mask, coor_x, coor_y, coor_z, region


def _get_deconv_pad_outpad(deconv_kernel):
    """Get padding and out padding for deconv layers."""
    if deconv_kernel == 4:
        padding = 1
        output_padding = 0
    elif deconv_kernel == 3:
        padding = 1
        output_padding = 1
    elif deconv_kernel == 2:
        padding = 0
        output_padding = 0
    else:
        raise ValueError(f"Not supported num_kernels ({deconv_kernel}).")

    return deconv_kernel, padding, output_padding
