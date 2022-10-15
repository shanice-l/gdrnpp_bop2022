import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
from mmcv.cnn import normal_init, constant_init
from timm.models.layers import StdConv2d

# from lib.torch_utils.layers.layer_utils import get_norm, get_nn_act_func
from lib.torch_utils.layers.conv_module import ConvModule


class ConvMaskXyzRegionHead(nn.Module):
    def __init__(
        self,
        in_dim,
        num_feat_layers=0,
        feat_dim=256,
        feat_kernel_size=3,
        use_ws=False,
        norm="GN",
        num_gn_groups=32,
        act="GELU",
        out_kernel_size=1,
        out_layer_shared=True,
        mask_num_classes=1,
        xyz_num_classes=1,
        region_num_classes=1,
        mask_out_dim=1,
        xyz_out_dim=3,
        region_out_dim=65,  # 64+1
    ):
        super().__init__()
        assert out_kernel_size in [
            1,
            3,
        ], "Only support output kernel size: 1 and 3"

        self.features = nn.ModuleList()
        for i in range(num_feat_layers):
            _in_dim = in_dim if i == 0 else feat_dim
            if use_ws:
                conv_cfg = dict(type="StdConv2d")
            else:
                conv_cfg = None
            self.features.append(
                ConvModule(
                    _in_dim,
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

        _in_dim = feat_dim if num_feat_layers > 0 else in_dim
        conv_layer = StdConv2d if use_ws else nn.Conv2d
        if self.out_layer_shared:
            out_dim = (
                self.mask_out_dim * self.mask_num_classes
                + self.xyz_out_dim * self.xyz_num_classes
                + self.region_out_dim * self.region_num_classes
            )
            self.out_layer = conv_layer(
                _in_dim,
                out_dim,
                kernel_size=out_kernel_size,
                padding=(out_kernel_size - 1) // 2,
                bias=True,
            )
        else:
            self.mask_out_layer = conv_layer(
                _in_dim,
                self.mask_out_dim * self.mask_num_classes,
                kernel_size=out_kernel_size,
                padding=(out_kernel_size - 1) // 2,
                bias=True,
            )
            self.xyz_out_layer = conv_layer(
                _in_dim,
                self.xyz_out_dim * self.xyz_num_classes,
                kernel_size=out_kernel_size,
                padding=(out_kernel_size - 1) // 2,
                bias=True,
            )
            self.region_out_layer = conv_layer(
                _in_dim,
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
            mask = out[:, :mask_dim, :, :]

            xyz_dim = self.xyz_out_dim * self.xyz_num_classes
            xyz = out[:, mask_dim : mask_dim + xyz_dim, :, :]

            region = out[:, mask_dim + xyz_dim :, :, :]

            bs, c, h, w = xyz.shape
            xyz = xyz.view(bs, 3, xyz_dim // 3, h, w)
            coor_x = xyz[:, 0, :, :, :]
            coor_y = xyz[:, 1, :, :, :]
            coor_z = xyz[:, 2, :, :, :]

        else:
            mask = self.mask_out_layer(x)

            xyz = self.xyz_out_layer(x)
            bs, c, h, w = xyz.shape
            xyz = xyz.view(bs, 3, c // 3, h, w)
            coor_x = xyz[:, 0, :, :, :]
            coor_y = xyz[:, 1, :, :, :]
            coor_z = xyz[:, 2, :, :, :]

            region = self.region_out_layer(x)
        return mask, coor_x, coor_y, coor_z, region
