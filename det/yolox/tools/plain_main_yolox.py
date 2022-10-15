#!/usr/bin/env python3
import logging
import os.path as osp

from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.engine import (
    default_argument_parser,
    launch,
)
from detectron2.engine.defaults import create_ddp_model
from detectron2.config import LazyConfig, instantiate

from lib.utils.setup_logger import setup_my_logger
import core.utils.my_comm as comm
from core.utils.my_checkpoint import MyCheckpointer
from det.yolox.engine.yolox_train_test_plain import do_train_yolox, do_test_yolox
from det.yolox.engine.yolox_setup import default_yolox_setup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("detectron2")


def setup(args):
    """Create configs and perform basic setups."""
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    default_yolox_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    if args.eval_only:  # eval
        model = instantiate(cfg.model)
        model.to(cfg.train.device)
        model = create_ddp_model(model)
        MyCheckpointer(model, save_dir=cfg.train.output_dir).resume_or_load(
            cfg.train.init_checkpoint, resume=args.resume
        )
        print(do_test_yolox(cfg, model))
    else:  # train
        do_train_yolox(args, cfg)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
