import copy
import time
import random
import numpy as np
import logging
import os
import os.path as osp
import sys
from typing import List, Mapping, Optional
import weakref
from collections import OrderedDict
from typing import Optional, Sequence

import core.utils.my_comm as comm
import detectron2.data.transforms as T
import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist
from core.utils.my_writer import MyPeriodicWriter
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
    is_parallel,
    save_checkpoint,
    setup_logger,
    synchronize,
)
from detectron2.utils.events import EventStorage, get_event_storage
from detectron2.config.instantiate import instantiate
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.build import AspectRatioGroupedDataset
from detectron2.engine import create_ddp_model, hooks
from detectron2.engine.train_loop import TrainerBase
from detectron2.evaluation import DatasetEvaluator, print_csv_format, verify_results
from detectron2.utils.collect_env import collect_env_info
from detectron2.utils.env import seed_all_rng

from fvcore.nn.precise_bn import get_bn_modules
from lib.utils.setup_logger import setup_my_logger, log_first_n
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel, DataParallel

from lib.torch_utils.solver.grad_clip_d2 import maybe_add_gradient_clipping
from lib.utils.config_utils import try_get_key
from core.utils.my_checkpoint import MyCheckpointer
from det.yolox.data import DataPrefetcher
from .yolox_inference import yolox_inference_on_dataset
from .yolox_setup import default_yolox_writers


logger = logging.getLogger(__name__)


class YOLOX_DefaultTrainer(TrainerBase):
    """A trainer with default training logic. It does the following:

    1. Create a :class:`SimpleTrainer` using model, optimizer, dataloader
       defined by the given config. Create a LR scheduler defined by the config.
    2. Load the last checkpoint or `cfg.train.init_checkpoint`, if exists, when
       `resume_or_load` is called.
    3. Register a few common hooks defined by the config.

    It is created to simplify the **standard model training workflow** and reduce code boilerplate
    for users who only need the standard training workflow, with standard features.
    It means this class makes *many assumptions* about your training logic that
    may easily become invalid in a new research. In fact, any assumptions beyond those made in the
    :class:`SimpleTrainer` are too much for research.

    The code of this class has been annotated about restrictive assumptions it makes.
    When they do not work for you, you're encouraged to:

    1. Overwrite methods of this class, OR:
    2. Use :class:`SimpleTrainer`, which only does minimal SGD training and
       nothing else. You can then add your own hooks if needed. OR:
    3. Write your own training loop similar to `tools/plain_train_net.py`.

    See the :doc:`/tutorials/training` tutorials for more details.

    Note that the behavior of this class, like other functions/classes in
    this file, is not stable, since it is meant to represent the "common default behavior".
    It is only guaranteed to work well with the standard models and training workflow in detectron2.
    To obtain more stable behavior, write your own training logic with other public APIs.

    Examples:
    ::
        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load()  # load last checkpoint or train.init_checkpoint
        trainer.train()

    Attributes:
        scheduler:
        checkpointer (DetectionCheckpointer):
        cfg (CfgNode):
    """

    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        """
        super().__init__()
        # if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
        #     setup_my_logger(name="det")

        # Assume these objects must be constructed in this order.
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)

        self.data_type = torch.float16 if cfg.train.amp.enabled else torch.float32

        train_loader_cfg = cfg.dataloader.train
        self.input_size = train_loader_cfg.dataset.img_size
        train_loader = instantiate(train_loader_cfg)

        # TODO: support train2 and train2_ratio
        ims_per_batch = train_loader_cfg.total_batch_size
        # only using train to determine iters_per_epoch
        if isinstance(train_loader, AspectRatioGroupedDataset):
            dataset_len = len(train_loader.dataset.dataset)
            iters_per_epoch = dataset_len // ims_per_batch
        else:
            dataset_len = len(train_loader.dataset)
            iters_per_epoch = dataset_len // ims_per_batch
        max_iter = cfg.train.total_epochs * iters_per_epoch
        cfg.train.iters_per_epoch = iters_per_epoch
        cfg.train.max_iter = max_iter
        cfg.train.no_aug_iters = cfg.train.no_aug_epochs * iters_per_epoch
        cfg.train.warmup_iters = cfg.train.warmup_epochs * iters_per_epoch
        logger.info("ims_per_batch: {}".format(ims_per_batch))
        logger.info("dataset length: {}".format(dataset_len))
        logger.info("iters per epoch: {}".format(iters_per_epoch))
        logger.info("total iters: {}".format(max_iter))

        cfg.train.eval_period = cfg.train.eval_period * cfg.train.iters_per_epoch
        cfg.train.checkpointer.period = cfg.train.checkpointer.period * cfg.train.iters_per_epoch

        OmegaConf.set_readonly(cfg, True)  # freeze config
        self.cfg = cfg

        model = create_ddp_model(model, broadcast_buffers=False)

        self.use_model_ema = cfg.train.ema
        if self.use_model_ema:
            self.ema_model = ModelEMA(model, 0.9998)

        amp_ckpt_data = {}
        self.init_model_loader_optimizer_amp(model, train_loader, optimizer, amp_ckpt_data)
        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        self.checkpointer = MyCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.train.output_dir,
            # trainer=weakref.proxy(self),
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            save_to_disk=comm.is_main_process(),
            **amp_ckpt_data,
        )
        self.start_iter = 0
        self.max_iter = cfg.train.max_iter

        self.register_hooks(self.build_hooks())

    def init_model_loader_optimizer_amp(self, model, train_loader, optimizer, amp_ckpt_data={}, train_loader2=None):
        amp_cfg = self.cfg.train.amp
        if amp_cfg.enabled:
            logger.info("Using pytorch amp")
            unsupported = "AMPTrainer does not support single-process multi-device training!"
            if isinstance(model, DistributedDataParallel):
                assert not (model.device_ids and len(model.device_ids) > 1), unsupported
            assert not isinstance(model, DataParallel), unsupported

        self.grad_scaler = GradScaler(enabled=amp_cfg.enabled)
        amp_ckpt_data["grad_scaler"] = self.grad_scaler
        self.init_model_loader_optimizer_simple(model, train_loader, optimizer, train_loader2=train_loader2)

    def init_model_loader_optimizer_simple(self, model, train_loader, optimizer, train_loader2=None):
        model.train()

        self.model = model

        self.data_loader = train_loader
        self._data_loader_iter = iter(train_loader)
        logger.info("init prefetcher, this might take one minute or less...")
        self.prefetcher = DataPrefetcher(train_loader)

        self.data_loader2 = train_loader2
        self._data_loader_iter2 = None
        self.prefecher2 = None
        if train_loader2 is not None:
            self._data_loader_iter2 = iter(train_loader2)
            logger.info("init prefetcher2, this might take one minute or less...")
            self.prefecher2 = DataPrefetcher(train_loader2)

        self.optimizer = optimizer

    def resume_or_load(self, resume=True):
        """NOTE: should run before train()
        if resume from a middle/last ckpt but want to reset iteration,
        remove the iteration key from the ckpt first
        """
        if resume:
            # NOTE: --resume always from last_checkpoint
            iter_saved = self.checkpointer.resume_or_load("", resume=True).get("iteration", -1)
        else:
            if self.cfg.train.resume_from != "":
                # resume_from a given ckpt
                iter_saved = self.checkpointer.load(self.cfg.train.resume_from).get("iteration", -1)
            else:
                # load from a given ckpt
                # iter_saved = self.checkpointer.load(self.cfg.train.init_checkpoint).get("iteration", -1)
                iter_saved = self.checkpointer.resume_or_load(self.cfg.train.init_checkpoint, resume=resume).get(
                    "iteration", -1
                )
        self.start_iter = iter_saved + 1

    def build_hooks(self):
        """Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        cfg = self.cfg
        train_loader_cfg = copy.deepcopy(cfg.dataloader.train)
        if OmegaConf.is_readonly(train_loader_cfg):
            OmegaConf.set_readonly(train_loader_cfg, False)

        train_loader_cfg.num_workers = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.train.eval_period,
                self.model,
                # Build a new data loader to not affect training
                instantiate(train_loader_cfg),
                cfg.test.precise_bn.num_iter,
            )
            if cfg.test.precise_bn.enabled and get_bn_modules(self.model)
            else None,
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(hooks.PeriodicCheckpointer(self.checkpointer, **cfg.train.checkpointer))

        def test_and_save_results():
            # TODO: check this ema
            if self.use_model_ema:
                evalmodel = self.ema_model.ema
            else:
                evalmodel = self.model
                if is_parallel(evalmodel):
                    evalmodel = evalmodel.module
            self._last_eval_results = self.test(self.cfg, evalmodel)
            return self._last_eval_results

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        ret.append(hooks.EvalHook(cfg.train.eval_period, test_and_save_results))

        if comm.is_main_process():
            # Here the default print/log frequency of each writer is used.
            # run writers in the end, so that evaluation metrics are written
            ret.append(MyPeriodicWriter(self.build_writers(), period=cfg.train.log_period))
        return ret

    def build_writers(self):
        """Build a list of writers to be used using :func:`default_writers()`.
        If you'd like a different list of writers, you can overwrite it in your
        trainer.

        Returns:
            list[EventWriter]: a list of :class:`EventWriter` objects.
        """
        return default_yolox_writers(self.cfg.train.output_dir, self.max_iter)

    def before_train(self):
        if try_get_key(self.cfg, "train.occupy_gpu", default=False):
            occupy_mem(comm.get_local_rank())
        super().before_train()  # for hooks

        if self.start_iter >= self.max_iter - self.cfg.train.no_aug_iters:
            self.close_mosaic()
            if self.cfg.train.use_l1 and self.cfg.train.l1_from_scratch is False:
                self.enable_l1()
            OmegaConf.set_readonly(self.cfg, False)
            logger.info(f"sync norm period changed from {self.cfg.train.sync_norm_period} to 1")
            self.cfg.train.sync_norm_period = 1  # sync norm every epoch when mosaic is closed
            OmegaConf.set_readonly(self.cfg, True)

        if self.cfg.train.use_l1 and self.cfg.train.l1_from_scratch is True:
            self.enable_l1()

        if self.use_model_ema:
            self.ema_model.updates = self.start_iter

    def train(self):
        """Run training.

        Returns:
            OrderedDict of results, if evaluation is enabled. Otherwise None.
        """
        logger.info(
            f"total batch size: {self.cfg.dataloader.train.total_batch_size}, num_gpus: {comm.get_world_size()}"
        )
        super().train(self.start_iter, self.max_iter)
        # if len(self.cfg.test.expected_results) and comm.is_main_process():
        #     assert hasattr(
        #         self, "_last_eval_results"
        #     ), "No evaluation results obtained during training!"
        #     verify_results(self.cfg, self._last_eval_results)
        #     return self._last_eval_results

    def enable_l1(self):
        logger.info("--->Add additional L1 loss now!")
        if comm.get_world_size() > 1:
            self.model.module.head.use_l1 = True
        else:
            self.model.head.use_l1 = True

    def close_mosaic(self):
        logger.info("--->No mosaic aug now!")
        self.data_loader.close_mosaic()

    def before_step(self):
        super().before_step()

        self.epoch = self.iter // self.cfg.train.iters_per_epoch
        if self.iter == self.max_iter - self.cfg.train.no_aug_iters:
            self.close_mosaic()
            if self.cfg.train.use_l1 and self.cfg.train.l1_from_scratch is False:
                self.enable_l1()
            OmegaConf.set_readonly(self.cfg, False)
            logger.info(f"sync norm period changed from {self.cfg.train.sync_norm_period} to 1")
            self.cfg.train.sync_norm_period = 1  # sync norm every epoch when mosaic is closed
            OmegaConf.set_readonly(self.cfg, True)
            if comm.is_main_process():
                self.checkpointer.save(
                    name=f"last_mosaic_epoch{self.epoch}_iter{self.iter}",
                    iteration=self.iter,
                )

    def run_step(self):
        assert self.model.training, "[YOLOX_DefaultTrainer] model was changed to eval mode!"

        # log_first_n(logging.INFO, f"running iter: {self.iter}", n=5)  # for debug

        start = time.perf_counter()  # get data --------------------------
        # inps, targets, _, _ = next(self._data_loader_iter)
        # inps, targets, scene_im_id, _, img_id = next(self._data_loader_iter)
        inps, targets = self.prefetcher.next()
        inps = inps.to(self.data_type)
        targets = targets.to(self.data_type)
        targets.requires_grad = False
        inps, targets = self.preprocess(inps, targets, self.input_size)
        data_time = time.perf_counter() - start

        with autocast(enabled=self.cfg.train.amp.enabled):
            out_dict, loss_dict = self.model(inps, targets)
            if isinstance(loss_dict, torch.Tensor):
                losses = loss_dict
                loss_dict = {"total_loss": loss_dict}
            else:
                losses = sum(loss_dict.values())

        # vis image
        # from det.yolox.utils.visualize import vis_train
        # vis_train(inps, targets, self.cfg)

        # optimizer step ------------------------------------------------
        self.optimizer.zero_grad()
        self.grad_scaler.scale(losses).backward()

        # write metrics before opt step
        self._write_metrics(loss_dict, data_time)

        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()

        if self.use_model_ema:
            self.ema_model.update(self.model)

        # log_first_n(logging.INFO, f"done iter: {self.iter}", n=5)  # for debug

    def after_step(self):
        for h in self._hooks:
            # NOTE: hack to save ema model
            if isinstance(h, hooks.PeriodicCheckpointer) and self.use_model_ema:
                h.checkpointer.model = self.ema_model.ema
            h.after_step()

        # sync norm
        if self.cfg.train.sync_norm_period > 0:
            if (self.epoch + 1) % self.cfg.train.sync_norm_period == 0:
                all_reduce_norm(self.model)

        # random resizing
        if self.cfg.train.random_size is not None and self.iter % 10 == 0:
            is_distributed = comm.get_world_size() > 1
            self.input_size = self.random_resize(self.data_loader, self.epoch, comm.get_rank(), is_distributed)

    def _write_metrics(
        self,
        loss_dict: Mapping[str, torch.Tensor],
        data_time: float,
        prefix: str = "",
    ) -> None:
        """
        Args:
            loss_dict (dict): dict of scalar losses
                losses should have a `loss` str;
                other scalar metrics can be also in it.
            data_time (float): time taken by the dataloader iteration
            prefix (str): prefix for logging keys
        """
        metrics_dict = {}
        for k, v in loss_dict.items():
            if isinstance(v, torch.Tensor):
                metrics_dict[k] = v.detach().cpu().item()
            else:
                metrics_dict[k] = v  # assume float/int
        metrics_dict["data_time"] = data_time

        # Gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in detectron2.
        all_metrics_dict = comm.gather(metrics_dict)

        if comm.is_main_process():
            storage = get_event_storage()

            # data_time among workers can have high variance. The actual latency
            # caused by data_time is the maximum among workers.
            data_time = np.max([x.pop("data_time") for x in all_metrics_dict])
            storage.put_scalar("data_time", data_time)

            storage.put_scalar("epoch", self.epoch)  # NOTE: added

            # average the rest metrics
            metrics_dict = {k: np.mean([x[k] for x in all_metrics_dict]) for k in all_metrics_dict[0].keys()}
            # NOTE: filter losses
            total_losses_reduced = sum([v for k, v in metrics_dict.items() if "loss" in k])
            if not np.isfinite(total_losses_reduced):
                raise FloatingPointError(
                    f"Loss became infinite or NaN at iteration={storage.iter}!\n" f"loss_dict = {metrics_dict}"
                )

            storage.put_scalar("{}total_loss".format(prefix), total_losses_reduced)
            if len(metrics_dict) > 1:
                storage.put_scalars(**metrics_dict)

    def preprocess(self, inputs, targets, tsize):
        scale_y = tsize[0] / self.input_size[0]
        scale_x = tsize[1] / self.input_size[1]
        if scale_x != 1 or scale_y != 1:
            inputs = nn.functional.interpolate(inputs, size=tsize, mode="bilinear", align_corners=False)
            targets[..., 1::2] = targets[..., 1::2] * scale_x
            targets[..., 2::2] = targets[..., 2::2] * scale_y
        return inputs, targets

    def random_resize(self, data_loader, epoch, rank, is_distributed):
        # randomly choose a int, *32, aspect ratio is the same as intput size
        tensor = torch.LongTensor(2).cuda()

        if rank == 0:
            size_factor = self.input_size[1] * 1.0 / self.input_size[0]  # w/h
            size = random.randint(*self.cfg.train.random_size)
            size = (int(32 * size), 32 * int(size * size_factor))
            tensor[0] = size[0]
            tensor[1] = size[1]

        if is_distributed:
            dist.barrier()
            dist.broadcast(tensor, 0)

        input_size = (tensor[0].item(), tensor[1].item())
        return input_size

    @classmethod
    def build_model(cls, cfg, verbose=True):
        """
        Returns:
            torch.nn.Module:

        It now calls :func:`detectron2.modeling.build_model`.
        Overwrite it if you'd like a different model.
        """
        model = instantiate(cfg.model)
        if verbose:
            logger.info("Model:\n{}".format(model))
        model.to(cfg.train.device)
        return model

    @classmethod
    def build_optimizer(cls, cfg, model):
        """
        Returns:
            torch.optim.Optimizer:
        """
        cfg.optimizer.params.model = model
        optimizer = instantiate(cfg.optimizer)
        optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        anneal_point = cfg.lr_config.get("anneal_point", 0)
        if cfg.train.anneal_after_warmup:
            anneal_point = min(
                anneal_point + cfg.train.warmup_epochs / (cfg.train.total_epochs - cfg.train.no_aug_epochs),
                1.0,
            )
        OmegaConf.set_readonly(cfg, False)
        cfg.lr_config.update(
            optimizer=optimizer,
            total_iters=cfg.train.max_iter - cfg.train.no_aug_iters,  # exclude no aug iters
            warmup_iters=cfg.train.warmup_iters,
            anneal_point=anneal_point,
        )
        OmegaConf.set_readonly(cfg, True)
        return instantiate(cfg.lr_config)

    @classmethod
    def test_single(cls, cfg, model, evaluator=None):
        """
        Args:
            cfg (CfgNode):
            model (nn.Module):
        Returns:
            dict: a dict of result metrics
        """
        test_dset_name = cfg.dataloader.test.dataset.lst.names
        if not isinstance(test_dset_name, str):
            test_dset_name = ",".join(test_dset_name)
        if OmegaConf.is_readonly(cfg):
            OmegaConf.set_readonly(cfg, False)
        cfg.dataloader.evaluator.output_dir = osp.join(cfg.train.output_dir, "inference", test_dset_name)
        OmegaConf.set_readonly(cfg, True)
        if evaluator is None:
            evaluator = instantiate(cfg.dataloader.evaluator)
        cls.auto_set_test_batch_size(cfg.dataloader.test)
        ret = yolox_inference_on_dataset(
            model,
            instantiate(cfg.dataloader.test),
            evaluator=evaluator,
            amp_test=cfg.test.amp_test,
            half_test=cfg.test.half_test,
            test_cfg=cfg.test,
            val_cfg=cfg.val,
        )
        if comm.is_main_process():
            assert isinstance(ret, dict), "Evaluator must return a dict on the main process. Got {} instead.".format(
                ret
            )
            logger.info("Evaluation results for {} in csv format:".format(test_dset_name))
            print_csv_format(ret)
        return ret

    @classmethod
    def test(cls, cfg, model, evaluators=None):
        """
        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                test_loaders
        Returns:
            dict: a dict of result metrics
        """
        loader_cfgs = cfg.dataloader.test
        if not isinstance(loader_cfgs, Sequence):
            return cls.test_single(cfg, model, evaluator=evaluators)

        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]

        if evaluators is not None:
            assert len(loader_cfgs) == len(evaluators), "{} != {}".format(len(loader_cfgs), len(evaluators))
        else:
            evaluator_cfgs = cfg.dataloader.evaluator
            assert isinstance(evaluator_cfgs, Sequence)
            assert len(loader_cfgs) == len(evaluator_cfgs), "{} != {}".format(len(loader_cfgs), len(evaluator_cfgs))

        results = OrderedDict()
        for idx, loader_cfg in enumerate(loader_cfgs):
            cls.auto_set_test_batch_size(loader_cfg)
            test_loader = instantiate(loader_cfg)

            test_dset_name = loader_cfg.dataset.lst.names
            if not isinstance(test_dset_name, str):
                test_dset_name = ",".join(test_dset_name)

            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    eval_cfg = evaluator_cfgs[idx]
                    if OmegaConf.is_readonly(eval_cfg):
                        OmegaConf.set_readonly(eval_cfg, False)
                    eval_cfg.output_dir = osp.join(cfg.train.output_dir, "inference", test_dset_name)
                    OmegaConf.set_readonly(eval_cfg, True)
                    evaluator = instantiate(eval_cfg)
                except NotImplementedError:
                    logger.warning("No evaluator found. Use `DefaultTrainer.test(evaluators=)` instead")
                    results[test_dset_name] = {}
                    continue
            ret_i = yolox_inference_on_dataset(
                model,
                test_loader,
                evaluator=evaluator,
                amp_test=cfg.test.amp_test,
                half_test=cfg.test.half_test,
                test_cfg=cfg.test,
                val_cfg=cfg.val,
            )
            results[test_dset_name] = ret_i
            if comm.is_main_process():
                assert isinstance(
                    ret_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(ret_i)
                logger.info("Evaluation results for {} in csv format:".format(test_dset_name))
                print_csv_format(ret_i)

        if len(results) == 1:
            results = list(results.values())[0]
        return results

    @classmethod
    def auto_set_test_batch_size(cls, loader_cfg):
        test_batch_size = loader_cfg.total_batch_size
        n_gpus = comm.get_world_size()
        if test_batch_size % n_gpus != 0:
            OmegaConf.set_readonly(loader_cfg, False)
            new_batch_size = int(np.ceil(test_batch_size / n_gpus) * n_gpus)
            loader_cfg.total_batch_size = new_batch_size
            logger.info(
                "test total batch size reset from {} to {}, n_gpus: {}".format(test_batch_size, new_batch_size, n_gpus)
            )
            OmegaConf.set_readonly(loader_cfg, True)
