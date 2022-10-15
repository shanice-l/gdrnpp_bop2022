import argparse

# from loguru import logger
import os
import os.path as osp
import sys
import weakref
from collections import OrderedDict
from typing import Optional
import torch
from fvcore.nn.precise_bn import get_bn_modules
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel

import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import CfgNode, LazyConfig
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.evaluation import (
    DatasetEvaluator,
    inference_on_dataset,
    print_csv_format,
    verify_results,
)
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer

from detectron2.utils.collect_env import collect_env_info
from detectron2.utils.env import seed_all_rng
from detectron2.utils.events import CommonMetricPrinter, JSONWriter, TensorboardXWriter
from detectron2.utils.file_io import PathManager

import mmcv
import PIL

from lib.utils.setup_logger import setup_my_logger
from lib.utils.setup_logger_loguru import setup_logger
from lib.utils.time_utils import get_time_str
from lib.utils.config_utils import try_get_key
import core.utils.my_comm as comm
from core.utils.my_writer import (
    MyCommonMetricPrinter,
    MyJSONWriter,
    MyTensorboardXWriter,
)


def _highlight(code, filename):
    try:
        import pygments
    except ImportError:
        return code

    from pygments.lexers import Python3Lexer, YamlLexer
    from pygments.formatters import Terminal256Formatter

    lexer = Python3Lexer() if filename.endswith(".py") else YamlLexer()
    code = pygments.highlight(code, lexer, Terminal256Formatter(style="monokai"))
    return code


def default_yolox_setup(cfg, args):
    """NOTE: compared to d2,
        1) logger has line number;
        2) more project related logger names;
        3) setup mmcv image backend
    Perform some basic common setups at the beginning of a job, including:

    1. Set up the detectron2 logger
    2. Log basic information about environment, cmdline arguments, and config
    3. Backup the config to the output directory

    Args:
        cfg (CfgNode or omegaconf.DictConfig): the full config to be used
        args (argparse.NameSpace): the command line arguments to be logged
    """
    output_dir = try_get_key(cfg, "OUTPUT_DIR", "output_dir", "train.output_dir")
    if comm.is_main_process() and output_dir:
        PathManager.mkdirs(output_dir)

    rank = comm.get_rank()
    # filename = osp.join(output_dir, f"log_{get_time_str()}.txt")
    # setup_logger(output_dir, distributed_rank=rank, filename=filename, mode="a")
    setup_my_logger(output_dir, distributed_rank=rank, name="fvcore")
    setup_my_logger(output_dir, distributed_rank=rank, name="mylib")
    setup_my_logger(output_dir, distributed_rank=rank, name="core")
    setup_my_logger(output_dir, distributed_rank=rank, name="det")
    setup_my_logger(output_dir, distributed_rank=rank, name="detectron2")
    setup_my_logger(output_dir, distributed_rank=rank, name="ref")
    setup_my_logger(output_dir, distributed_rank=rank, name="tests")
    setup_my_logger(output_dir, distributed_rank=rank, name="tools")
    setup_my_logger(output_dir, distributed_rank=rank, name="__main__")
    logger = setup_my_logger(output_dir, distributed_rank=rank, name=__name__)

    logger.info("Rank of current process: {}. World size: {}".format(rank, comm.get_world_size()))
    logger.info("Environment info:\n" + collect_env_info())

    logger.info("Command line arguments: " + str(args))
    if hasattr(args, "config_file") and args.config_file != "":
        logger.info(
            "Contents of args.config_file={}:\n{}".format(
                args.config_file,
                _highlight(PathManager.open(args.config_file, "r").read(), args.config_file),
            )
        )

    if comm.is_main_process() and output_dir:
        # Note: some of our scripts may expect the existence of
        # config.yaml in output directory
        path = os.path.join(output_dir, "config.yaml")
        if isinstance(cfg, CfgNode):
            logger.info("Running with full config:\n{}".format(_highlight(cfg.dump(), ".yaml")))
            with PathManager.open(path, "w") as f:
                f.write(cfg.dump())
        else:
            LazyConfig.save(cfg, path)
        logger.info("Full config saved to {}".format(path))

    # make sure each worker has a different, yet deterministic seed if specified
    seed = try_get_key(cfg, "SEED", "train.seed", default=-1)
    logger.info(f"seed: {seed}")
    seed_all_rng(None if seed < 0 else seed + rank)

    cudnn_deterministic = try_get_key(cfg, "CUDNN_DETERMINISTIC", "train.cudnn_deterministic", default=False)
    if cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
        logger.warning(
            "You have turned on the CUDNN deterministic setting, "
            "which can slow down your training considerably! You may see unexpected behavior "
            "when restarting from checkpoints."
        )

    # cudnn benchmark has large overhead. It shouldn't be used considering the small size of
    # typical validation set.
    if not (hasattr(args, "eval_only") and args.eval_only):  # currently only used for train
        torch.backends.cudnn.benchmark = try_get_key(cfg, "CUDNN_BENCHMARK", "train.cudnn_benchmark", default=False)

    # set mmcv backend
    mmcv_backend = try_get_key(cfg, "mmcv_backend", "MMCV_BACKEND", default="cv2")
    if mmcv_backend == "pillow" and "post" not in PIL.__version__:
        logger.warning("Consider installing pillow-simd!")
    logger.info(f"Used mmcv backend: {mmcv_backend}")
    mmcv.use_backend(mmcv_backend)


def default_yolox_writers(output_dir: str, max_iter: Optional[int] = None, backup=True):
    """
    Build a list of :class:`EventWriter` to be used.
    It now consists of a :class:`CommonMetricPrinter`,
    :class:`TensorboardXWriter` and :class:`JSONWriter`.

    Args:
        output_dir: directory to store JSON metrics and tensorboard events
        max_iter: the total number of iterations

    Returns:
        list[EventWriter]: a list of :class:`EventWriter` objects.
    """
    tb_logdir = osp.join(output_dir, "tb")
    mmcv.mkdir_or_exist(tb_logdir)
    if backup and comm.is_main_process():
        old_tb_logdir = osp.join(output_dir, "tb_old")
        mmcv.mkdir_or_exist(old_tb_logdir)
        os.system("mv -v {} {}".format(osp.join(tb_logdir, "events.*"), old_tb_logdir))
    return [
        # It may not always print what you want to see, since it prints "common" metrics only.
        MyCommonMetricPrinter(max_iter),
        MyJSONWriter(os.path.join(output_dir, "metrics.json")),
        MyTensorboardXWriter(tb_logdir, backend="tensorboardx"),
    ]
