# TODO: just use plain train loop
import time
import os.path as osp
import logging
from collections import OrderedDict
from collections.abc import Sequence
from detectron2.engine import (
    SimpleTrainer,
    default_writers,
    hooks,
)
from detectron2.data.build import AspectRatioGroupedDataset
from detectron2.data import MetadataCatalog
from detectron2.utils.events import EventStorage
from detectron2.config import LazyConfig, instantiate
from detectron2.evaluation import print_csv_format
from detectron2.config import LazyConfig, instantiate
from detectron2.engine import (
    AMPTrainer,
    SimpleTrainer,
    # default_writers,
    hooks,
)
from detectron2.engine.defaults import create_ddp_model
from detectron2.evaluation import print_csv_format

import core.utils.my_comm as comm
from core.utils.my_writer import MyPeriodicWriter
from core.utils.my_checkpoint import MyCheckpointer
from det.yolox.data import DataPrefetcher
from det.yolox.utils import (
    MeterBuffer,
    ModelEMA,
    all_reduce_norm,
    get_model_info,
    get_rank,
    get_world_size,
    gpu_mem_usage,
    load_ckpt,
    occupy_mem,
    save_checkpoint,
    setup_logger,
    synchronize,
)
from .yolox_inference import yolox_inference_on_dataset
from .yolox_setup import default_yolox_writers


logger = logging.getLogger(__name__)


def do_test_yolox(cfg, model, use_all_reduce_norm=False):
    if "evaluator" not in cfg.dataloader:
        logger.warning("no evaluator in cfg.dataloader, do nothing!")
        return

    if use_all_reduce_norm:
        all_reduce_norm(model)

    if not isinstance(cfg.dataloader.test, Sequence):
        test_dset_name = cfg.dataloader.test.dataset.lst.names
        if not isinstance(test_dset_name, str):
            test_dset_name = ",".join(test_dset_name)
        cfg.dataloader.evaluator.output_dir = osp.join(cfg.train.output_dir, "inference", test_dset_name)
        ret = yolox_inference_on_dataset(
            model,
            instantiate(cfg.dataloader.test),
            evaluator=instantiate(cfg.dataloader.evaluator),
            amp_test=cfg.test.amp_test,
            half_test=cfg.test.half_test,
            test_cfg=cfg.test,
        )
        logger.info("Evaluation results for {} in csv format:".format(test_dset_name))
        print_csv_format(ret)
        return ret
    else:
        results = OrderedDict()
        for loader_cfg, eval_cfg in zip(cfg.dataloader.test, cfg.dataloader.evaluator):
            test_dset_name = loader_cfg.dataset.lst.names
            if not isinstance(test_dset_name, str):
                test_dset_name = ",".join(test_dset_name)
            eval_cfg.output_dir = osp.join(cfg.train.output_dir, "inference", test_dset_name)
            ret_i = yolox_inference_on_dataset(
                model,
                instantiate(loader_cfg),
                evaluator=instantiate(eval_cfg),
                amp_test=cfg.test.amp_test,
                half_test=cfg.test.half_test,
                test_cfg=cfg.test,
            )
            logger.info("Evaluation results for {} in csv format:".format(test_dset_name))
            print_csv_format(ret_i)
            results[test_dset_name] = ret_i
        return results


def do_train_yolox(args, cfg):
    """
    Args:
        cfg: an object with the following attributes:
            model: instantiate to a module
            dataloader.{train,test}: instantiate to dataloaders
            dataloader.evaluator: instantiate to evaluator for test set
            optimizer: instantaite to an optimizer
            lr_multiplier: instantiate to a fvcore scheduler
            train: other misc config defined in `configs/common/train.py`, including:
                output_dir (str)
                init_checkpoint (str)
                amp.enabled (bool)
                max_iter (int)
                eval_period, log_period (int)
                device (str)
                checkpointer (dict)
                ddp (dict)
    """
    model = instantiate(cfg.model)
    logger.info("Model:\n{}".format(model))
    model.to(cfg.train.device)

    cfg.optimizer.params.model = model
    optim = instantiate(cfg.optimizer)

    # TODO: support train2 and train2_ratio
    train_loader = instantiate(cfg.dataloader.train)
    ims_per_batch = cfg.dataloader.train.total_batch_size
    # only using train to determine iters_per_epoch
    if isinstance(train_loader, AspectRatioGroupedDataset):
        dataset_len = len(train_loader.dataset.dataset)
        iters_per_epoch = dataset_len // ims_per_batch
    else:
        dataset_len = len(train_loader.dataset)
        iters_per_epoch = dataset_len // ims_per_batch
    max_iter = cfg.lr_config.total_epochs * iters_per_epoch
    cfg.train.max_iter = max_iter
    cfg.train.no_aug_iters = cfg.train.no_aug_epochs * iters_per_epoch
    cfg.train.warmup_iters = cfg.train.warmup_epochs * iters_per_epoch
    logger.info("ims_per_batch: {}".format(ims_per_batch))
    logger.info("dataset length: {}".format(dataset_len))
    logger.info("iters per epoch: {}".format(iters_per_epoch))
    logger.info("total iters: {}".format(max_iter))

    anneal_point = cfg.lr_config.get("anneal_point", 0)
    if cfg.train.anneal_after_warmup:
        anneal_point = min(
            anneal_point + cfg.train.warmup_epochs / (cfg.train.total_epochs - cfg.train.no_aug_epochs),
            1.0,
        )
    cfg.lr_config.update(
        optimizer=optim,
        total_iters=max_iter - cfg.train.no_aug_iters,  # exclude no aug iters
        warmup_iters=cfg.train.warmup_iters,
        anneal_point=anneal_point,
    )

    model = create_ddp_model(model, **cfg.train.ddp)
    trainer = (AMPTrainer if cfg.train.amp.enabled else SimpleTrainer)(model, train_loader, optim)
    checkpointer = MyCheckpointer(
        model,
        cfg.train.output_dir,
        optimizer=optim,
        trainer=trainer,
        save_to_disk=comm.is_main_process(),
    )
    trainer.register_hooks(
        [
            hooks.IterationTimer(),
            hooks.LRScheduler(scheduler=instantiate(cfg.lr_config)),
            hooks.PeriodicCheckpointer(checkpointer, **cfg.train.checkpointer) if comm.is_main_process() else None,
            hooks.EvalHook(cfg.train.eval_period, lambda: do_test_yolox(cfg, model)),
            MyPeriodicWriter(
                default_yolox_writers(cfg.train.output_dir, cfg.train.max_iter),
                period=cfg.train.log_period,
            )
            if comm.is_main_process()
            else None,
        ]
    )

    checkpointer.resume_or_load(cfg.train.init_checkpoint, resume=args.resume)
    if args.resume and checkpointer.has_checkpoint():
        # The checkpoint stores the training iteration that just finished, thus we start
        # at the next iteration
        start_iter = trainer.iter + 1
    else:
        start_iter = 0
    trainer.train(start_iter, cfg.train.max_iter)
