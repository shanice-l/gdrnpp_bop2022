"""modified from mmcv.cnn, pass norm and act from outside to support more
types."""
import inspect
import warnings

import torch.nn as nn

from mmcv.cnn.utils import constant_init, kaiming_init
from mmcv.cnn.bricks.conv import CONV_LAYERS
from mmcv.cnn.bricks.conv import build_conv_layer as build_conv_layer_mmcv
from mmcv.cnn.bricks.padding import build_padding_layer
from timm.models.layers import StdConv2d
from .mean_conv_deconv import MeanConv2d
from .layer_utils import get_norm, get_nn_act_func, infer_abbr


def build_conv_layer(cfg, *args, **kwargs):
    """Build convolution layer.

    Args:
        cfg (None or dict): The conv layer config, which should contain:
            - type (str): Layer type.
            - layer args: Args needed to instantiate an conv layer.
        args (argument list): Arguments passed to the `__init__`
            method of the corresponding conv layer.
        kwargs (keyword arguments): Keyword arguments passed to the `__init__`
            method of the corresponding conv layer.

    Returns:
        nn.Module: Created conv layer.
    """
    if cfg is None:
        cfg_ = dict(type="Conv2d")
    else:
        if not isinstance(cfg, dict):
            raise TypeError("cfg must be a dict")
        if "type" not in cfg:
            raise KeyError('the cfg dict must contain the key "type"')
        cfg_ = cfg.copy()

    layer_type = cfg_.pop("type")
    if layer_type == "StdConv2d":
        CONV_LAYERS.register_module()(StdConv2d)
    elif layer_type == "MeanConv2d":
        CONV_LAYERS.register_module()(MeanConv2d)

    if layer_type not in CONV_LAYERS:
        raise KeyError(f"Unrecognized norm type {layer_type}")
    else:
        conv_layer = CONV_LAYERS.get(layer_type)

    layer = conv_layer(*args, **kwargs, **cfg_)

    return layer


class ConvModule(nn.Module):
    """A conv block that bundles conv/norm/activation layers.

    This block simplifies the usage of convolution layers, which are commonly
    used with a norm layer (e.g., BatchNorm/GroupNorm) and activation layer (e.g., ReLU/LeakyReLU/GELU/SiLU/Mish).
    conv layer is based on: `build_conv_layer()`,

    Besides, we add some additional features in this module.
    1. Automatically set `bias` of the conv layer.
    2. Spectral norm is supported.
    3. More padding modes are supported. Before PyTorch 1.5, nn.Conv2d only
    supports zero and circular padding, and we add "reflect" padding mode.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int | tuple[int]): Same as nn.Conv2d.
        stride (int | tuple[int]): Same as nn.Conv2d.
        padding (int | tuple[int]): Same as nn.Conv2d.
        dilation (int | tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool | str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if `norm_cfg` is None, otherwise
            False. Default: "auto".
        conv_cfg (dict): Config dict for convolution layer. Default: None,
            which means using conv2d.
        norm: normalization layer, str_name or None (default).
        act: activation layer, str_name or None.
        inplace (bool): Whether to use inplace mode for activation.
            Default: True.
        with_spectral_norm (bool): Whether use spectral norm in conv module.
            Default: False.
        padding_mode (str): If the `padding_mode` has not been supported by
            current `Conv2d` in PyTorch, we will use our own padding layer
            instead. Currently, we support ['zeros', 'circular'] with official
            implementation and ['reflect'] with our own implementation.
            Default: 'zeros'.
        order (tuple[str]): The order of conv/norm/activation layers. It is a
            sequence of "conv", "norm" and "act". Common examples are
            ("conv", "norm", "act") and ("act", "conv", "norm").
            Default: ('conv', 'norm', 'act').
    """

    _abbr_ = "conv_block"

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias="auto",
        conv_cfg=None,
        norm=None,
        num_gn_groups=32,
        act=None,
        inplace=True,
        with_spectral_norm=False,
        padding_mode="zeros",
        order=("conv", "norm", "act"),
    ):
        super(ConvModule, self).__init__()
        assert conv_cfg is None or isinstance(conv_cfg, dict)
        official_padding_mode = ["zeros", "circular"]
        self.conv_cfg = conv_cfg
        # modified
        self.with_norm = self._is_not_none(norm)
        self.with_activation = self._is_not_none(act)
        self.with_spectral_norm = with_spectral_norm
        self.with_explicit_padding = padding_mode not in official_padding_mode
        self.order = order
        assert isinstance(self.order, tuple) and len(self.order) == 3
        assert set(order) == set(["conv", "norm", "act"])

        # if the conv layer is before a norm layer, bias is unnecessary.
        if bias == "auto":
            bias = not self.with_norm
        self.with_bias = bias

        if self.with_norm and self.with_bias:
            warnings.warn("ConvModule has norm and bias at the same time")

        if self.with_explicit_padding:
            pad_cfg = dict(type=padding_mode)
            self.padding_layer = build_padding_layer(pad_cfg, padding)

        # reset padding to 0 for conv module
        conv_padding = 0 if self.with_explicit_padding else padding
        # build convolution layer
        self.conv = build_conv_layer(
            conv_cfg,
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=conv_padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        # export the attributes of self.conv to a higher level for convenience
        self.in_channels = self.conv.in_channels
        self.out_channels = self.conv.out_channels
        self.kernel_size = self.conv.kernel_size
        self.stride = self.conv.stride
        self.padding = padding
        self.dilation = self.conv.dilation
        self.transposed = self.conv.transposed
        self.output_padding = self.conv.output_padding
        self.groups = self.conv.groups

        if self.with_spectral_norm:
            self.conv = nn.utils.spectral_norm(self.conv)

        # build normalization layers
        if self.with_norm:
            # norm layer is after conv layer
            if order.index("norm") > order.index("conv"):
                norm_channels = out_channels
            else:
                norm_channels = in_channels
            self.norm = get_norm(norm, norm_channels, num_gn_groups=num_gn_groups)
            self.norm_name = infer_abbr(type(self.norm))  # type(self.norm).__name__.lower()
            self.add_module(self.norm_name, self.norm)

        # build activation layer
        if self.with_activation:
            self.activate = get_nn_act_func(act, inplace=inplace)

        # Use msra init by default
        self.init_weights()

    def _is_not_none(self, v):
        if v is None:
            return False
        if isinstance(v, str):
            if len(v) == 0 or v.lower() == "none":
                return False
        return True

    @property
    def norm(self):
        return getattr(self, self.norm_name)

    def init_weights(self):
        # 1. It is mainly for customized conv layers with their own
        #    initialization manners, and we do not want ConvModule to
        #    overrides the initialization.
        # 2. For customized conv layers without their own initialization
        #    manners, they will be initialized by this method with default
        #    `kaiming_init`.
        # 3. For PyTorch's conv layers, they will be initialized anyway by
        #    their own `reset_parameters` methods.
        if not hasattr(self.conv, "init_weights"):  # do not override if a custom conv layer has initialized
            if self.with_activation and infer_abbr(type(self.activate)) == "leakyrelu":
                nonlinearity = "leaky_relu"
                a = self.activate.negative_slope  # 0.1
            else:
                nonlinearity = "relu"
                a = 0
            kaiming_init(self.conv, a=a, nonlinearity=nonlinearity)
        if self.with_norm:
            constant_init(self.norm, 1, bias=0)

    def forward(self, x, activate=True, norm=True):
        for layer in self.order:
            if layer == "conv":
                if self.with_explicit_padding:
                    x = self.padding_layer(x)
                x = self.conv(x)
            elif layer == "norm" and norm and self.with_norm:
                x = self.norm(x)
            elif layer == "act" and activate and self.with_activation:
                x = self.activate(x)
        return x
