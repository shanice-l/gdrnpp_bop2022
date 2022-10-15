import inspect
import warnings
import torch
from torch import nn
from torch.nn.modules.instancenorm import _InstanceNorm
from torch.nn.modules.batchnorm import _BatchNorm
import torch.nn.functional as F
from detectron2.layers.batch_norm import BatchNorm2d, FrozenBatchNorm2d, NaiveSyncBatchNorm
from detectron2.utils import env
from .acon import AconC, MetaAconC


class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, x):
        return x * self.scale


class SMU(nn.Module):
    def __init__(self, alpha=0.25, mu_init=1e6):
        super().__init__()
        self.alpha = alpha
        self.mu = nn.Parameter(torch.FloatTensor([mu_init]))

    def forward(self, x):
        return 0.5 * ((1 + self.alpha) * x + (1 - self.alpha) * x * torch.erf(self.mu * (1 - self.alpha) * x))


def get_norm(norm, out_channels, num_gn_groups=32):
    """
    Args:
        norm (str or callable): either one of BN, SyncBN, FrozenBN, GN;
            or a callable that takes a channel number and returns
            the normalization layer as a nn.Module.
    Returns:
        nn.Module or nn.Identity(): the normalization layer
    """
    if norm is None:
        return nn.Identity()
    if isinstance(norm, str):
        if len(norm) == 0 or norm.lower() == "none":
            return nn.Identity()
        norm = {
            "BN": BatchNorm2d,
            # Fixed in https://github.com/pytorch/pytorch/pull/36382
            "SyncBN": NaiveSyncBatchNorm if env.TORCH_VERSION <= (1, 5) else nn.SyncBatchNorm,
            "FrozenBN": FrozenBatchNorm2d,
            "GN": lambda channels: nn.GroupNorm(num_gn_groups, channels),
            "IN": nn.InstanceNorm2d,
            # for debugging:
            "nnSyncBN": nn.SyncBatchNorm,
            "naiveSyncBN": NaiveSyncBatchNorm,
        }[norm]
    return norm(out_channels)


def get_nn_act_func(act, inplace=True, **kwargs):
    """Using torch.nn if possible."""
    if act is None:
        return nn.Identity()

    if act.lower() == "relu":
        act_func = nn.ReLU(inplace=inplace)
    elif act.lower() == "sigmoid":
        act_func = nn.Sigmoid()
    elif act.lower() == "prelu":
        # num_parameters=1, init=0.25
        act_func = nn.PReLU(**kwargs)
    elif act.lower() in ["lrelu", "leaky_relu", "leakyrelu"]:
        kwargs.setdefault("negative_slope", 0.1)
        act_func = nn.LeakyReLU(inplace=inplace, **kwargs)
    elif act.lower() in ["silu", "swish"]:
        # requires pytorch>=1.7
        act_func = nn.SiLU(inplace=inplace)
    elif act.lower() in ["aconc"]:
        act_func = AconC(**kwargs)
    elif act.lower() in ["metaaconc"]:
        act_func = MetaAconC(**kwargs)
    elif act.lower() == "gelu":
        act_func = nn.GELU()
    elif act.lower() == "mish":
        act_func = nn.Mish(inplace=inplace)
    elif act.lower() == "smu":
        alpha = kwargs.get("alpha", 0.25)
        mu_init = kwargs.get("mu_init", 1e6)
        act_func = SMU(alpha=alpha, mu_init=mu_init)
    elif len(act) == 0 or act.lower() == "none":
        return nn.Identity()
    else:
        raise ValueError(f"Unknown activation function: {act}.")
    return act_func


def soft_argmax(x, beta=1000.0, dim=1, base_index=0, step_size=1, keepdim=False):
    """Compute the forward pass of the soft arg-max function as defined below:

    SoftArgMax(x) = \sum_i (i * softmax(x)_i)
    :param x: The input to the soft arg-max layer
    :return: Output of the soft arg-max layer
    """
    smax = F.softmax(x * beta, dim=dim)
    end_index = base_index + x.shape[dim] * step_size
    indices = torch.arange(start=base_index, end=end_index, step=step_size).to(x)
    view_shape = [1 for _ in x.shape]
    view_shape[dim] = x.shape[dim]
    indices = indices.view(view_shape)
    return torch.sum(smax * indices, dim=dim, keepdim=keepdim)


def gumbel_soft_argmax(
    x,
    tau=1.0,
    dim=1,
    hard=True,
    eps=1e-10,
    base_index=0,
    step_size=1,
    keepdim=False,
):
    """
    NOTE: this is stochastic
    """
    gsmax = F.gumbel_softmax(x, tau=tau, dim=dim, hard=hard, eps=eps)
    end_index = base_index + x.shape[dim] * step_size
    indices = torch.arange(start=base_index, end=end_index, step=step_size).to(x)
    view_shape = [1 for _ in x.shape]
    view_shape[dim] = x.shape[dim]
    indices = indices.view(view_shape)
    return torch.sum(gsmax * indices, dim=dim, keepdim=keepdim)


def infer_abbr(class_type):
    """Infer abbreviation from the class name.

    When we build a norm layer with `build_norm_layer()`, we want to preserve
    the norm type in variable names, e.g, self.bn1, self.gn. This method will
    infer the abbreviation to map class types to abbreviations.

    Rule 1: If the class has the property "_abbr_", return the property.
    Rule 2: If the parent class is _BatchNorm, GroupNorm, LayerNorm or
    InstanceNorm, the abbreviation of this layer will be "bn", "gn", "ln" and
    "in" respectively.
    Rule 3: If the class name contains "batch", "group", "layer" or "instance",
    the abbreviation of this layer will be "bn", "gn", "ln" and "in"
    respectively.
    Rule 4: Otherwise, the abbreviation falls back to "class_type.__name__.lower()".

    Args:
        class_type (type): The norm layer type.

    Returns:
        str: The inferred abbreviation.
    """
    if not inspect.isclass(class_type):
        raise TypeError(f"class_type must be a type, but got {type(class_type)}")
    if hasattr(class_type, "_abbr_"):
        return class_type._abbr_
    if issubclass(class_type, _InstanceNorm):  # IN is a subclass of BN
        return "in"
    elif issubclass(class_type, _BatchNorm):
        return "bn"
    elif issubclass(class_type, nn.GroupNorm):
        return "gn"
    elif issubclass(class_type, nn.LayerNorm):
        return "ln"
    else:
        class_name = class_type.__name__.lower()
        if "batch" in class_name:
            return "bn"
        elif "group" in class_name:
            return "gn"
        elif "layer" in class_name:
            return "ln"
        elif "instance" in class_name:
            return "in"
        else:
            return class_name


def resize(
    input,
    size=None,
    scale_factor=None,
    mode="nearest",
    align_corners=None,
    warning=True,
):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if (
                    (output_h > 1 and output_w > 1 and input_h > 1 and input_w > 1)
                    and (output_h - 1) % (input_h - 1)
                    and (output_w - 1) % (input_w - 1)
                ):
                    warnings.warn(
                        f"When align_corners={align_corners}, "
                        "the output would more aligned if "
                        f"input size {(input_h, input_w)} is `x+1` and "
                        f"out size {(output_h, output_w)} is `nx+1`"
                    )
    if isinstance(size, torch.Size):
        size = tuple(int(x) for x in size)
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class Upsample(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode="nearest", align_corners=None):
        super(Upsample, self).__init__()
        self.size = size
        if isinstance(scale_factor, tuple):
            self.scale_factor = tuple(float(factor) for factor in scale_factor)
        else:
            self.scale_factor = float(scale_factor) if scale_factor else None
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        if not self.size:
            size = [int(t * self.scale_factor) for t in x.shape[-2:]]
        else:
            size = self.size
        return resize(x, size, None, self.mode, self.align_corners)
