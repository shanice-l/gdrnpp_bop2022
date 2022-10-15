# Copyright (c) Facebook, Inc. and its affiliates.
import datetime
import logging
import time
from collections import OrderedDict, abc
from contextlib import ExitStack, contextmanager
from typing import List, Union

from tqdm import tqdm
from omegaconf import OmegaConf
import torch
from torch import nn
from torch.cuda.amp import autocast

from detectron2.utils.logger import log_every_n_seconds
from detectron2.evaluation import DatasetEvaluator, DatasetEvaluators, inference_context

from core.utils.my_comm import get_world_size, is_main_process
from det.yolox.utils import (
    gather,
    postprocess,
    synchronize,
    time_synchronized,
    xyxy2xywh,
)

logger = logging.getLogger(__name__)


def yolox_inference_on_dataset(
    model,
    data_loader,
    evaluator: Union[DatasetEvaluator, List[DatasetEvaluator], None],
    amp_test=False,
    half_test=False,
    trt_file=None,
    decoder=None,
    test_cfg=OmegaConf.create(
        dict(
            test_size=(640, 640),
            conf_thr=0.01,
            nms_thr=0.65,
            num_classes=80,
        )
    ),
    val_cfg=OmegaConf.create(
        dict(
            eval_cached=False,
        )
    ),
):
    """Run model on the data_loader and evaluate the metrics with evaluator.
    Also benchmark the inference speed of `model.__call__` accurately. The
    model will be used in eval mode.

    Args:
        model (callable): a callable which takes an object from
            `data_loader` and returns some outputs.

            If it's an nn.Module, it will be temporarily set to `eval` mode.
            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator: the evaluator(s) to run. Use `None` if you only want to benchmark,
            but don't want to do any evaluation.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    num_devices = get_world_size()
    assert int(half_test) + int(amp_test) <= 1, "half_test and amp_test cannot both be set"
    logger.info(f"half_test: {half_test}, amp_test: {amp_test}")

    logger.info("Start inference on {} batches".format(len(data_loader)))

    cfg = test_cfg

    total = len(data_loader)  # inference data loader must have a fixed length
    if evaluator is None:
        # create a no-op evaluator
        evaluator = DatasetEvaluators([])
    if isinstance(evaluator, abc.MutableSequence):
        evaluator = DatasetEvaluators(evaluator)

    if val_cfg.get("eval_cached", False):
        results = evaluator.evaluate(eval_cached=True)
        # An evaluator may return None when not in main process.
        # Replace it by an empty dict instead to make it easier for downstream code to handle
        if results is None:
            results = {}
        return results

    evaluator.reset()
    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_data_time = 0
    total_compute_time = 0
    total_nms_time = 0
    total_eval_time = 0
    iters_record = 0
    augment = cfg.get("augment", False)
    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())

        tensor_type = torch.cuda.HalfTensor if (half_test or amp_test) else torch.cuda.FloatTensor
        if half_test:
            model = model.half()

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, cfg.test_size[0], cfg.test_size[1]).cuda()
            model(x)
            model = model_trt

        progress_bar = tqdm if is_main_process() else iter

        start_data_time = time_synchronized()
        for idx, inputs in enumerate(progress_bar(data_loader)):
            imgs, _, scene_im_ids, info_imgs, ids = inputs
            imgs = imgs.type(tensor_type)

            compute_time = 0

            # skip the the last iters since batchsize might be not enough for batch inference
            # is_time_record = idx < len(data_loader) - 1
            is_time_record = idx < len(data_loader)

            total_data_time += time.perf_counter() - start_data_time
            if idx == num_warmup and is_time_record:
                start_time = time.perf_counter()
                total_data_time = 0
                total_compute_time = 0
                total_nms_time = 0
                total_eval_time = 0
                iters_record = 0

            if is_time_record:
                start_compute_time = time.perf_counter()

            if trt_file is not None:
                det_preds = model(imgs)
                outputs = {"det_preds": det_preds}
            else:
                # outputs = model(imgs)
                outputs = model(imgs, augment=augment, cfg=cfg)

            if decoder is not None:
                outputs["det_preds"] = decoder(outputs["det_preds"], dtype=outputs.type())
            if is_time_record:
                infer_end_time = time_synchronized()
                total_compute_time += infer_end_time - start_compute_time

            # import ipdb; ipdb.set_trace()
            outputs["det_preds"] = postprocess(outputs["det_preds"], cfg.num_classes, cfg.conf_thr, cfg.nms_thr)
            if is_time_record:
                nms_end_time = time_synchronized()
                total_nms_time += nms_end_time - infer_end_time
                compute_time = nms_end_time - start_compute_time
                outputs["time"] = compute_time

            evaluator.process(outputs, scene_im_ids, info_imgs, ids, cfg)
            if is_time_record:
                total_eval_time += time.perf_counter() - nms_end_time
                # iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
                iters_record += 1

            data_ms_per_iter = total_data_time / iters_record * 1000
            compute_ms_per_iter = total_compute_time / iters_record * 1000
            nms_ms_per_iter = total_nms_time / iters_record * 1000
            eval_ms_per_iter = total_eval_time / iters_record * 1000
            total_ms_per_iter = (time.perf_counter() - start_time) / iters_record * 1000
            if idx >= num_warmup * 2 or compute_ms_per_iter > 5000:
                eta = datetime.timedelta(seconds=int(total_ms_per_iter / 1000 * (total - idx - 1)))
                log_every_n_seconds(
                    logging.WARN,
                    (
                        f"Inference done {idx + 1}/{total}. "
                        f"Dataloading: {data_ms_per_iter:.4f} ms/iter. "
                        f"Inference: {compute_ms_per_iter:.4f} ms/iter. "
                        f"NMS: {nms_ms_per_iter:.4f} ms/iter. "
                        f"Eval: {eval_ms_per_iter:.4f} ms/iter. "
                        f"Total: {total_ms_per_iter:.4f} ms/iter. "
                        f"ETA={eta}"
                    ),
                    n=5,
                    name=__name__,
                )
            start_data_time = time.perf_counter()

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))

    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.3f} ms / iter per device, on {} devices)".format(
            total_time_str, total_time * 1000 / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.3f} ms / iter per device, on {} devices)".format(
            total_compute_time_str,
            total_compute_time * 1000 / (total - num_warmup),
            num_devices,
        )
    )
    total_nms_time_str = str(datetime.timedelta(seconds=int(total_nms_time)))
    logger.info(
        "Total inference nms time: {} ({:.3f} ms / iter per device, on {} devices)".format(
            total_nms_time_str,
            total_nms_time * 1000 / (total - num_warmup),
            num_devices,
        )
    )

    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results
