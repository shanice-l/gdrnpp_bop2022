# Copyright (c) Facebook, Inc. and its affiliates.
# modified to support full_model gradient norm clip
import copy
import itertools
import logging
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Type, Union
from omegaconf.omegaconf import OmegaConf

import torch
from detectron2.config import CfgNode
from fvcore.common.param_scheduler import CosineParamScheduler, MultiStepParamScheduler
from lib.utils.config_utils import try_get_key


_GradientClipperInput = Union[torch.Tensor, Iterable[torch.Tensor]]
_GradientClipper = Callable[[_GradientClipperInput], None]


class GradientClipType(Enum):
    VALUE = "value"
    NORM = "norm"
    FULL_MODEL = "full_model"


def _create_gradient_clipper(cfg) -> _GradientClipper:
    """Creates gradient clipping closure to clip by value or by norm, according
    to the provided config."""
    cfg = copy.deepcopy(cfg)

    _clip_value = try_get_key(cfg, "CLIP_VALUE", "clip_value", default=1.0)
    _norm_type = try_get_key(cfg, "NORM_TYPE", "norm_type", default=2.0)

    def clip_grad_norm(p: _GradientClipperInput):
        torch.nn.utils.clip_grad_norm_(p, _clip_value, _norm_type)

    def clip_grad_value(p: _GradientClipperInput):
        torch.nn.utils.clip_grad_value_(p, _clip_value)

    _GRADIENT_CLIP_TYPE_TO_CLIPPER = {
        GradientClipType.VALUE: clip_grad_value,
        GradientClipType.NORM: clip_grad_norm,
        GradientClipType.FULL_MODEL: clip_grad_norm,
    }
    _clip_type = try_get_key(cfg, "CLIP_TYPE", "clip_type", default="full_model")
    return _GRADIENT_CLIP_TYPE_TO_CLIPPER[GradientClipType(_clip_type)]


def _generate_optimizer_class_with_gradient_clipping(
    optimizer: Type[torch.optim.Optimizer],
    *,
    per_param_clipper: Optional[_GradientClipper] = None,
    global_clipper: Optional[_GradientClipper] = None,
) -> Type[torch.optim.Optimizer]:
    """Dynamically creates a new type that inherits the type of a given
    instance and overrides the `step` method to add gradient clipping."""
    assert (
        per_param_clipper is None or global_clipper is None
    ), "Not allowed to use both per-parameter clipping and global clipping"

    def optimizer_wgc_step(self, closure=None):
        if per_param_clipper is not None:
            for group in self.param_groups:
                for p in group["params"]:
                    per_param_clipper(p)
        else:
            # global clipper for future use with detr
            # (https://github.com/facebookresearch/detr/pull/287)
            all_params = itertools.chain(*[g["params"] for g in self.param_groups])
            global_clipper(all_params)
        super(type(self), self).step(closure)

    OptimizerWithGradientClip = type(
        optimizer.__name__ + "WithGradientClip",
        (optimizer,),
        {"step": optimizer_wgc_step},
    )
    return OptimizerWithGradientClip


def maybe_add_gradient_clipping(cfg: CfgNode, optimizer: Type[torch.optim.Optimizer]) -> Type[torch.optim.Optimizer]:
    """If gradient clipping is enabled through config options, wraps the
    existing optimizer type to become a new dynamically created class
    OptimizerWithGradientClip that inherits the given optimizer and overrides
    the `step` method to include gradient clipping.

    Args:
        cfg: CfgNode, configuration options
        optimizer: type. A subclass of torch.optim.Optimizer
    Return:
        type: either the input `optimizer` (if gradient clipping is disabled), or
            a subclass of it with gradient clipping included in the `step` method.
    """
    clip_cfg = try_get_key(
        cfg, "SOLVER.CLIP_GRADIENTS", "train.grad_clip", default=OmegaConf.create(dict(enabled=False))
    )
    if not try_get_key(clip_cfg, "ENABLED", "enabled", default=False):
        return optimizer
    if isinstance(optimizer, torch.optim.Optimizer):
        optimizer_type = type(optimizer)
    else:
        assert issubclass(optimizer, torch.optim.Optimizer), optimizer
        optimizer_type = optimizer

    grad_clipper = _create_gradient_clipper(clip_cfg)
    _clip_type = try_get_key(clip_cfg, "CLIP_TYPE", "clip_type", default="full_model")
    if _clip_type != "full_model":
        OptimizerWithGradientClip = _generate_optimizer_class_with_gradient_clipping(
            optimizer_type, per_param_clipper=grad_clipper
        )
    else:
        OptimizerWithGradientClip = _generate_optimizer_class_with_gradient_clipping(
            optimizer_type, global_clipper=grad_clipper
        )
    if isinstance(optimizer, torch.optim.Optimizer):
        optimizer.__class__ = OptimizerWithGradientClip  # a bit hacky, not recommended
        return optimizer
    else:
        return OptimizerWithGradientClip
