import os.path as osp

import torch
from detectron2.config import LazyCall as L
from detectron2.solver.build import get_default_optimizer_params

from .yolox_base import train, val, test, model, dataloader, optimizer, lr_config, DATASETS  # noqa
from det.yolox.data import build_yolox_test_loader, ValTransform
from det.yolox.data.datasets import Base_DatasetFromList
from detectron2.data import get_detection_dataset_dicts
from det.yolox.evaluators import YOLOX_COCOEvaluator
from lib.torch_utils.solver.ranger import Ranger

train.update(
    output_dir=osp.abspath(__file__).replace("configs", "output", 1)[0:-3],
    exp_name=osp.split(osp.abspath(__file__))[1][0:-3],  # .py
)
train.amp.enabled = True

model.backbone.depth = 1.33
model.backbone.width = 1.25

model.head.num_classes = 28

train.init_checkpoint = "pretrained_models/yolox/yolox_x.pth"

# datasets
DATASETS.TRAIN = ["itodd_pbr_train"]
DATASETS.TEST = ["itodd_bop_test"]

dataloader.train.dataset.lst.names = DATASETS.TRAIN
dataloader.train.total_batch_size = 32

# color aug
dataloader.train.aug_wrapper.COLOR_AUG_PROB = 0.8
dataloader.train.aug_wrapper.COLOR_AUG_TYPE = "code"
dataloader.train.aug_wrapper.COLOR_AUG_CODE = (
    "Sequential(["
    # Sometimes(0.5, PerspectiveTransform(0.05)),
    # Sometimes(0.5, CropAndPad(percent=(-0.05, 0.1))),
    # Sometimes(0.5, Affine(scale=(1.0, 1.2))),
    "Sometimes(0.5, CoarseDropout( p=0.2, size_percent=0.05) ),"
    "Sometimes(0.4, GaussianBlur((0., 3.))),"
    "Sometimes(0.3, pillike.EnhanceSharpness(factor=(0., 50.))),"
    "Sometimes(0.3, pillike.EnhanceContrast(factor=(0.2, 50.))),"
    "Sometimes(0.5, pillike.EnhanceBrightness(factor=(0.1, 6.))),"
    "Sometimes(0.3, pillike.EnhanceColor(factor=(0., 20.))),"
    "Sometimes(0.5, Add((-25, 25), per_channel=0.3)),"
    "Sometimes(0.3, Invert(0.2, per_channel=True)),"
    "Sometimes(0.5, Multiply((0.6, 1.4), per_channel=0.5)),"
    "Sometimes(0.5, Multiply((0.6, 1.4))),"
    "Sometimes(0.1, AdditiveGaussianNoise(scale=10, per_channel=True)),"
    "Sometimes(0.5, iaa.contrast.LinearContrast((0.5, 2.2), per_channel=0.3)),"
    # "Sometimes(0.5, Grayscale(alpha=(0.0, 1.0))),"  # maybe remove for det
    "], random_order=True)"
    # cosy+aae
)

# hsv color aug
dataloader.train.aug_wrapper.AUG_HSV_PROB = 1.0
dataloader.train.aug_wrapper.HSV_H = 0.015
dataloader.train.aug_wrapper.HSV_S = 0.7
dataloader.train.aug_wrapper.HSV_V = 0.4
dataloader.train.aug_wrapper.FORMAT = "RGB"

optimizer = L(Ranger)(
    params=L(get_default_optimizer_params)(
        # params.model is meant to be set to the model object, before instantiating
        # the optimizer.
        weight_decay_norm=0.0,
        weight_decay_bias=0.0,
    ),
    lr=0.001,  # bs=64
    # momentum=0.9,
    weight_decay=0,
    # nesterov=True,
)

train.total_epochs = 30
train.no_aug_epochs = 15
train.checkpointer = dict(period=2, max_to_keep=10)

test.test_dataset_names = DATASETS.TEST
test.augment = True
test.scales = (1, 0.75, 0.83, 1.12, 1.25)
test.conf_thr = 0.001

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
        total_batch_size=1,
        # total_batch_size=64,
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
