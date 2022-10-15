from itertools import count
import os
import os.path as osp
from omegaconf import OmegaConf

import torch
import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from detectron2.data import get_detection_dataset_dicts
from detectron2.solver.build import get_default_optimizer_params

# import torch.nn as nn

from det.yolox.models import YOLOX, YOLOPAFPN, YOLOXHead
from det.yolox.data import (
    # COCODataset,
    TrainTransform,
    ValTransform,
    # YoloBatchSampler,
    # DataLoader,
    # InfiniteSampler,
    MosaicDetection,
    build_yolox_train_loader,
    build_yolox_test_loader,
)
from det.yolox.data.datasets import Base_DatasetFromList
from det.yolox.utils import LRScheduler

# from detectron2.evaluation import COCOEvaluator
# from det.yolox.evaluators import COCOEvaluator
from det.yolox.evaluators import YOLOX_COCOEvaluator
from lib.torch_utils.solver.lr_scheduler import flat_and_anneal_lr_scheduler


# Common training-related configs that are designed for "tools/lazyconfig_train_net.py"
# You can use your own instead, together with your own train_net.py
train = dict(
    # NOTE: need to copy these two lines to get correct dirs
    output_dir=osp.abspath(__file__).replace("configs", "output", 1)[0:-3],
    exp_name=osp.split(osp.abspath(__file__))[1][0:-3],  # .py
    seed=-1,
    cudnn_deterministic=False,
    cudnn_benchmark=True,
    init_checkpoint="",
    # init_checkpoint="pretrained_models/yolox/yolox_s.pth",
    resume_from="",
    # init_checkpoint="detectron2://ImageNetPretrained/MSRA/R-50.pkl",
    # max_iter=90000,
    amp=dict(  # options for Automatic Mixed Precision
        enabled=True,
    ),
    grad_clip=dict(  # options for grad clipping
        enabled=False,
        clip_type="full_model",  # value, norm, full_model
        clip_value=1.0,
        norm_type=2.0,
    ),
    ddp=dict(  # options for DistributedDataParallel
        broadcast_buffers=False,
        find_unused_parameters=False,
        fp16_compression=False,
    ),
    # NOTE: epoch based period
    checkpointer=dict(period=1, max_to_keep=10),  # options for PeriodicCheckpointer
    # eval_period=5000,
    eval_period=-1,  # epoch based
    log_period=20,
    device="cuda",
    # ...
    basic_lr_per_img=0.01 / 64.0,  # 1.5625e-4
    random_size=(14, 26),  # set None to disable; randomly choose a int in this range, and *32
    mscale=(0.8, 1.6),
    ema=True,
    total_epochs=16,
    warmup_epochs=5,
    no_aug_epochs=2,
    sync_norm_period=10,  # sync norm every n epochs
    # l1 loss:
    # 1) if use_l1 and l1_from_sctrach: use l1 for the whole training phase
    # 2) use_l1=False: no l1 at all
    # 3) use_l1 and l1_from_scratch=False: just use l1 after closing mosaic (YOLOX default)
    l1_from_scratch=False,
    use_l1=True,
    anneal_after_warmup=True,
    # ...
    occupy_gpu=False,
)
train = OmegaConf.create(train)


# OmegaConf.register_new_resolver(
#      "mul2", lambda x: x*2
# )

# --------------------------------------------------------------------
# model
# --------------------------------------------------------------------
model = L(YOLOX)(
    backbone=L(YOLOPAFPN)(
        depth=1.0,
        width=1.0,
        in_channels=[256, 512, 1024],
    ),
    head=L(YOLOXHead)(
        num_classes=1,
        width="${..backbone.width}",
        # width="${mul2: ${..backbone.width}}",  # NOTE: do not forget $
        in_channels="${..backbone.in_channels}",
    ),
)

# --------------------------------------------------------------------
# optimizer
# --------------------------------------------------------------------
optimizer = L(torch.optim.SGD)(
    params=L(get_default_optimizer_params)(
        # params.model is meant to be set to the model object, before instantiating
        # the optimizer.
        weight_decay_norm=0.0,
        weight_decay_bias=0.0,
    ),
    lr=0.01,  # bs=64
    momentum=0.9,
    weight_decay=5e-4,
    nesterov=True,
)


lr_config = L(flat_and_anneal_lr_scheduler)(
    warmup_method="pow",
    warmup_pow=2,
    warmup_factor=0.0,
    # to be set
    # optimizer=
    # total_iters=total_iters,  # to be set
    # warmup_iters=epoch_len * 3,
    # anneal_point=5 / (total_epochs - 15),
    anneal_method="cosine",
    target_lr_factor=0.05,
)


DATASETS = dict(TRAIN=("",), TEST=("",))
DATASETS = OmegaConf.create(DATASETS)


dataloader = OmegaConf.create()
dataloader.train = L(build_yolox_train_loader)(
    dataset=L(Base_DatasetFromList)(
        split="train",
        lst=L(get_detection_dataset_dicts)(names=DATASETS.TRAIN),
        img_size=(640, 640),
        preproc=L(TrainTransform)(
            max_labels=50,
        ),
    ),
    aug_wrapper=L(MosaicDetection)(
        mosaic=True,
        img_size="${..dataset.img_size}",
        preproc=L(TrainTransform)(
            max_labels=120,
        ),
        degrees=10.0,
        translate=0.1,
        mosaic_scale=(0.1, 2),
        mixup_scale=(0.5, 1.5),
        shear=2.0,
        enable_mixup=True,
        mosaic_prob=1.0,
        mixup_prob=1.0,
    ),
    # reference_batch_size=64,
    total_batch_size=64,  # 8x8gpu
    num_workers=4,
    pin_memory=True,
)


val = dict(
    eval_cached=False,
)
val = OmegaConf.create(val)


test = dict(
    test_dataset_names=DATASETS.TEST,
    test_size=(640, 640),  # (height, width)
    conf_thr=0.01,
    nms_thr=0.65,
    num_classes="${model.head.num_classes}",
    amp_test=False,
    half_test=True,
    precise_bn=dict(
        enabled=False,
        num_iter=200,
    ),
    # fuse_conv_bn=False,
    fuse_conv_bn=True,
)
test = OmegaConf.create(test)


# NOTE: for multiple test loaders, just write it as a list
dataloader.test = [
    L(build_yolox_test_loader)(
        dataset=L(Base_DatasetFromList)(
            split="test",
            lst=L(get_detection_dataset_dicts)(names=test_dataset_name, filter_empty=False),
            img_size="${test.test_size}",
            preproc=L(ValTransform)(
                legacy=False,
            ),
        ),
        # total_batch_size=1,
        total_batch_size=64,
        num_workers=4,
        pin_memory=True,
    )
    for test_dataset_name in test.test_dataset_names
]


dataloader.evaluator = [
    L(YOLOX_COCOEvaluator)(
        dataset_name=test_dataset_name,
        filter_scene=False,
    )
    for test_dataset_name in test.test_dataset_names
]
