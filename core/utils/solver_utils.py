# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
from typing import Any, Dict, List
import logging
import torch

from detectron2.config import CfgNode

from lib.torch_utils.solver.lr_scheduler import flat_and_anneal_lr_scheduler
from lib.torch_utils.solver.optimize import _get_optimizer
from lib.torch_utils.solver.grad_clip_d2 import maybe_add_gradient_clipping
from mmcv.runner.optimizer import (
    OPTIMIZERS,
    DefaultOptimizerConstructor,
    build_optimizer,
)
from mmcv.utils import build_from_cfg


__all__ = [
    "my_build_optimizer",
    "build_optimizer_d2",
    "build_lr_scheduler",
    "build_optimizer_with_params",
]


def register_optimizer(name):
    """TODO: add more optimizers"""
    if name in OPTIMIZERS:
        return
    if name == "Ranger":
        from lib.torch_utils.solver.ranger import Ranger

        # from lib.torch_utils.solver.ranger2020 import Ranger
        OPTIMIZERS.register_module()(Ranger)

    elif name == "Ranger21":
        from lib.torch_utils.solver.ranger21 import Ranger21

        OPTIMIZERS.register_module()(Ranger21)
    elif name == "Lamb":
        from timm.optim import Lamb

        OPTIMIZERS.register_module()(Lamb)
    elif name == "MADGRAD":
        from lib.torch_utils.solver.madgrad import MADGRAD

        OPTIMIZERS.register_module()(MADGRAD)
    elif name == "NAdamW":
        from lib.torch_utils.solver.nadamw import NAdamW

        OPTIMIZERS.register_module()(NAdamW)
    elif name in ["AdaBelief", "RangerAdaBelief"]:
        from lib.torch_utils.solver.AdaBelief import AdaBelief
        from lib.torch_utils.solver.ranger_adabelief import RangerAdaBelief

        OPTIMIZERS.register_module()(AdaBelief)
        OPTIMIZERS.register_module()(RangerAdaBelief)
    elif name in ["SGDP", "AdamP"]:
        from lib.torch_utils.solver.adamp import AdamP
        from lib.torch_utils.solver.sgdp import SGDP

        OPTIMIZERS.register_module()(AdamP)
        OPTIMIZERS.register_module()(SGDP)
    elif name in ["SGD_GC", "SGD_GCC"]:
        from lib.torch_utils.solver.sgd_gc import SGD_GC, SGD_GCC

        OPTIMIZERS.register_module()(SGD_GC)
        OPTIMIZERS.register_module()(SGD_GCC)
    else:
        raise ValueError(f"Unknown optimizer name: {name}")


def build_optimizer_with_params(cfg, params):
    if cfg.SOLVER.OPTIMIZER_CFG == "":
        raise RuntimeError("please provide cfg.SOLVER.OPTIMIZER_CFG to build optimizer")
    if isinstance(cfg.SOLVER.OPTIMIZER_CFG, str):
        optim_cfg = eval(cfg.SOLVER.OPTIMIZER_CFG)
    else:
        optim_cfg = cfg.SOLVER.OPTIMIZER_CFG
    optim_cfg = copy.deepcopy(optim_cfg)  # avoid adding params to cfg
    register_optimizer(optim_cfg["type"])

    optim_cfg["params"] = params
    optimizer = build_from_cfg(optim_cfg, OPTIMIZERS)
    return maybe_add_gradient_clipping(cfg, optimizer)


def my_build_optimizer(cfg: CfgNode, model: torch.nn.Module) -> torch.optim.Optimizer:
    """Build an optimizer from config."""
    if cfg.SOLVER.OPTIMIZER_CFG != "":
        optim_cfg = eval(cfg.SOLVER.OPTIMIZER_CFG)
        register_optimizer(optim_cfg["type"])
        optimizer = build_optimizer(model, optim_cfg)
    else:
        # otherwise use this d2 builder
        optimizer = build_optimizer_d2(cfg, model)
    return maybe_add_gradient_clipping(cfg, optimizer)


def build_optimizer_d2(cfg: CfgNode, model: torch.nn.Module) -> torch.optim.Optimizer:
    """Build an optimizer from config.

    (Call my_build_optimizer instead)
    """
    params: List[Dict[str, Any]] = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if key.endswith("norm.weight") or key.endswith("norm.bias"):
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_NORM
        elif key.endswith(".bias"):
            # NOTE: unlike Detectron v1, we now default BIAS_LR_FACTOR to 1.0
            # and WEIGHT_DECAY_BIAS to WEIGHT_DECAY so that bias optimizer
            # hyperparameters are by default exactly the same as for regular
            # weights.
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    optim_cfg = dict(type=cfg.SOLVER.get("OPTIMIZER_NAME", "SGD"), lr=cfg.SOLVER.BASE_LR)
    solver_name = cfg.SOLVER.get("OPTIMIZER_NAME", "SGD").lower()
    if solver_name in ["sgd", "rmsprop"]:
        optim_cfg["momentum"] = cfg.SOLVER.MOMENTUM
    # TODO: more kwargs for other optimizer types
    optimizer = _get_optimizer(params, optim_cfg, use_hvd=False)
    # optimizer = torch.optim.SGD(params, lr, momentum=cfg.SOLVER.MOMENTUM)
    return optimizer


def build_lr_scheduler(
    cfg: CfgNode,
    optimizer: torch.optim.Optimizer,
    total_iters: int,
    return_function: bool = False,
) -> torch.optim.lr_scheduler._LRScheduler:
    """Build a LR scheduler from config."""
    name = cfg.SOLVER.LR_SCHEDULER_NAME
    if name.lower() == "flat_and_anneal":
        return flat_and_anneal_lr_scheduler(
            optimizer,
            total_iters=total_iters,  # NOTE: TOTAL_EPOCHS * len(train_loader)
            warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.WARMUP_METHOD,  # default "linear"
            anneal_method=cfg.SOLVER.ANNEAL_METHOD,
            anneal_point=cfg.SOLVER.ANNEAL_POINT,  # default 0.72
            steps=cfg.SOLVER.get("REL_STEPS", [2 / 3.0, 8 / 9.0]),  # default [2/3., 8/9.], relative decay steps
            target_lr_factor=cfg.SOLVER.get("TARGET_LR_FACTOR", 0),
            poly_power=cfg.SOLVER.get("POLY_POWER", 1.0),
            step_gamma=cfg.SOLVER.GAMMA,  # default 0.1
            return_function=return_function,
        )

    # attempt to use detectron2's schedulers
    from fvcore.common.param_scheduler import (
        CosineParamScheduler,
        MultiStepParamScheduler,
    )
    from detectron2.solver.lr_scheduler import (
        LRMultiplier,
        WarmupParamScheduler,
    )

    if name == "WarmupMultiStepLR":
        # convert relative steps to absolute steps
        steps = [rel_step * total_iters for rel_step in cfg.SOLVER.REL_STEPS if rel_step <= 1]
        if len(steps) != len(cfg.SOLVER.REL_STEPS):
            logger = logging.getLogger(__name__)
            logger.warning("SOLVER.REL_STEPS contains values larger than 1. " "These values will be ignored.")
        sched = MultiStepParamScheduler(
            values=[cfg.SOLVER.GAMMA**k for k in range(len(steps) + 1)],
            milestones=steps,
            num_updates=total_iters,
        )
    elif name == "WarmupCosineLR":
        # TODO: we can use the composite scheduler to construct warmup_flat_cosine schedule
        sched = CosineParamScheduler(1, 0)
    else:
        raise ValueError("Unknown LR scheduler: {}".format(name))

    sched = WarmupParamScheduler(
        sched,
        cfg.SOLVER.WARMUP_FACTOR,
        cfg.SOLVER.WARMUP_ITERS / total_iters,
        cfg.SOLVER.WARMUP_METHOD,
    )
    return LRMultiplier(optimizer, multiplier=sched, max_iter=total_iters)
