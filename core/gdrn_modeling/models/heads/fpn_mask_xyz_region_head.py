from abc import ABCMeta
import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
from mmcv.cnn import normal_init, constant_init

from lib.torch_utils.layers.layer_utils import resize
from lib.torch_utils.layers.conv_module import ConvModule


class BaseDecodeHead(nn.Module, metaclass=ABCMeta):
    """Base class for BaseDecodeHead.

    Args:
        in_channels (int|Sequence[int]): Input channels.
        channels (int): Channels after modules, before conv_seg.
        num_classes (int): Number of classes.
        dropout_ratio (float): Ratio of dropout layer. Default: 0.1.
        conv_cfg (dict|None): Config of conv layers. Default: None.
        in_index (int|Sequence[int]): Input feature index. Default: -1
        input_transform (str|None): Transformation type of input features.
            Options: 'resize_concat', 'multiple_select', None.
            'resize_concat': Multiple feature maps will be resize to the
                same size as first one and than concat together.
                Usually used in FCN head of HRNet.
            'multiple_select': Multiple feature maps will be bundle into
                a list and passed into decode head.
            None: Only one select feature map is allowed.
            Default: None.
        ignore_index (int | None): The label index to be ignored. When using
            masked BCE loss, ignore_index should be set to None. Default: 255
        sampler (dict|None): The config of segmentation map sampler.
            Default: None.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
    """

    def __init__(
        self,
        in_channels,
        channels,
        *,
        dropout_ratio=0.1,
        conv_cfg=None,
        norm=None,
        act="relu",
        in_index=-1,
        input_transform=None,
        align_corners=False,
    ):
        super(BaseDecodeHead, self).__init__()
        self._init_inputs(in_channels, in_index, input_transform)
        self.channels = channels
        self.dropout_ratio = dropout_ratio
        self.conv_cfg = conv_cfg
        self.norm = norm
        self.act = act
        self.in_index = in_index
        self.align_corners = align_corners

        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None
        self.fp16_enabled = False

    def extra_repr(self):
        """Extra repr."""
        s = f"input_transform={self.input_transform}, " f"align_corners={self.align_corners}"
        return s

    def _init_inputs(self, in_channels, in_index, input_transform):
        """Check and initialize input transforms.

        The in_channels, in_index and input_transform must match.
        Specifically, when input_transform is None, only single feature map
        will be selected. So in_channels and in_index must be of type int.
        When input_transform
        Args:
            in_channels (int|Sequence[int]): Input channels.
            in_index (int|Sequence[int]): Input feature index.
            input_transform (str|None): Transformation type of input features.
                Options: 'resize_concat', 'multiple_select', None.
                'resize_concat': Multiple feature maps will be resize to the
                    same size as first one and than concat together.
                    Usually used in FCN head of HRNet.
                'multiple_select': Multiple feature maps will be bundle into
                    a list and passed into decode head.
                None: Only one select feature map is allowed.
        """

        if input_transform is not None:
            assert input_transform in ["resize_concat", "multiple_select"]
        self.input_transform = input_transform
        self.in_index = in_index
        if input_transform is not None:
            assert isinstance(in_channels, (list, tuple))
            assert isinstance(in_index, (list, tuple))
            assert len(in_channels) == len(in_index)
            if input_transform == "resize_concat":
                self.in_channels = sum(in_channels)
            else:
                self.in_channels = in_channels
        else:
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            self.in_channels = in_channels

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor]): List of multi-level img features.
        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == "resize_concat":
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode="bilinear",
                    align_corners=self.align_corners,
                )
                for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == "multiple_select":
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs


class FPNMaskXyzRegionHead(BaseDecodeHead):
    """Panoptic Feature Pyramid Networks. This head is the implementation of
    `Semantic FPN.

    <https://arxiv.org/abs/1901.02446>`_.
    Args:
        feature_strides (tuple[int]): The strides for input feature maps.
            stack_lateral. All strides suppose to be power of 2. The first
            one is of largest resolution.
    """

    def __init__(
        self,
        feature_strides,
        out_kernel_size=1,
        out_layer_shared=True,
        mask_num_classes=1,
        xyz_num_classes=1,
        region_num_classes=1,
        mask_out_dim=1,
        xyz_out_dim=3,
        region_out_dim=65,  # 64+1,
        **kwargs,
    ):
        super().__init__(input_transform="multiple_select", **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        self.scale_heads = nn.ModuleList()
        for i in range(len(feature_strides)):
            head_length = max(
                1,
                int(np.log2(feature_strides[i]) - np.log2(feature_strides[0])),
            )
            scale_head = []
            for k in range(head_length):
                scale_head.append(
                    ConvModule(
                        self.in_channels[i] if k == 0 else self.channels,
                        self.channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm=self.norm,
                        act=self.act,
                    )
                )
                if feature_strides[i] != feature_strides[0]:
                    scale_head.append(
                        nn.Upsample(
                            scale_factor=2,
                            mode="bilinear",
                            align_corners=self.align_corners,
                        )
                    )
            self.scale_heads.append(nn.Sequential(*scale_head))

        self.out_layer_shared = out_layer_shared
        self.mask_num_classes = mask_num_classes
        self.xyz_num_classes = xyz_num_classes
        self.region_num_classes = region_num_classes

        self.mask_out_dim = mask_out_dim
        self.xyz_out_dim = xyz_out_dim
        self.region_out_dim = region_out_dim

        _in_dim = self.channels
        if self.out_layer_shared:
            out_dim = (
                self.mask_out_dim * self.mask_num_classes
                + self.xyz_out_dim * self.xyz_num_classes
                + self.region_out_dim * self.region_num_classes
            )
            self.out_layer = nn.Conv2d(
                _in_dim,
                out_dim,
                kernel_size=out_kernel_size,
                padding=(out_kernel_size - 1) // 2,
                bias=True,
            )
        else:
            self.mask_out_layer = nn.Conv2d(
                _in_dim,
                self.mask_out_dim * self.mask_num_classes,
                kernel_size=out_kernel_size,
                padding=(out_kernel_size - 1) // 2,
                bias=True,
            )
            self.xyz_out_layer = nn.Conv2d(
                _in_dim,
                self.xyz_out_dim * self.xyz_num_classes,
                kernel_size=out_kernel_size,
                padding=(out_kernel_size - 1) // 2,
                bias=True,
            )
            self.region_out_layer = nn.Conv2d(
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

    def forward(self, inputs):

        x = self._transform_inputs(inputs)  # strides: [4, 8, 16, 32]

        output = self.scale_heads[0](x[0])
        for i in range(1, len(self.feature_strides)):
            # non inplace
            output = output + resize(
                self.scale_heads[i](x[i]),
                size=output.shape[2:],
                mode="bilinear",
                align_corners=self.align_corners,
            )

        output = self.get_output(output)
        return output

    def get_output(self, x):
        if self.dropout is not None:
            x = self.dropout(x)
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
