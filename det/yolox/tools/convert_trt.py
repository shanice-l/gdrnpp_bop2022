#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import os.path as osp
import sys
import logging
from loguru import logger

import tensorrt as trt
import torch
from torch2trt import torch2trt

from detectron2.engine import default_argument_parser
from detectron2.config import LazyConfig, instantiate

cur_dir = osp.abspath(osp.dirname(__file__))
sys.path.insert(0, osp.join(cur_dir, "../../../"))
from core.utils.my_checkpoint import MyCheckpointer
from det.yolox.engine.yolox_setup import default_yolox_setup
from det.yolox.engine.yolox_trainer import YOLOX_DefaultTrainer


def setup(args):
    """Create configs and perform basic setups."""
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    default_yolox_setup(cfg, args)
    return cfg


@logger.catch
def main(args):
    cfg = setup(args)
    Trainer = YOLOX_DefaultTrainer
    model = Trainer.build_model(cfg)

    ckpt_file = args.ckpt
    MyCheckpointer(model).load(ckpt_file)
    logger.info("loaded checkpoint done.")

    model.eval()
    model.head.decode_in_inference = False
    x = torch.ones(1, 3, cfg.test.test_size[0], cfg.test.test_size[1]).cuda()
    model_trt = torch2trt(
        model,
        [x],
        fp16_mode=True,
        log_level=trt.Logger.INFO,
        max_workspace_size=(1 << 32),
    )

    filename_wo_ext, ext = osp.splitext(ckpt_file)
    trt_file = filename_wo_ext + "_trt" + ext
    torch.save(model_trt.state_dict(), trt_file)
    logger.info("Converted TensorRT model done.")

    engine_file = filename_wo_ext + "_trt.engine"
    with open(engine_file, "wb") as f:
        f.write(model_trt.engine.serialize())
    logger.info("Converted TensorRT model engine file is saved for C++ inference.")


if __name__ == "__main__":
    """python det/yolox/tools/convert_trt.py --config-file <path/to/cfg.py>

    --ckpt <path/to/ckpt.pth>
    """
    parser = default_argument_parser()
    parser.add_argument("--ckpt", type=str, help="ckpt path")
    args = parser.parse_args()
    main(args)
