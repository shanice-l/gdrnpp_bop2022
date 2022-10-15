#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch
import torch.nn as nn
from torch.nn.modules import loss

from .yolo_head import YOLOXHead
from .yolo_pafpn import YOLOPAFPN
from det.yolox.utils.model_utils import scale_img


class YOLOX(nn.Module):
    """YOLOX model module.

    The module list is defined by create_yolov3_modules function. The
    network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(self, backbone=None, head=None):
        super().__init__()
        if backbone is None:
            backbone = YOLOPAFPN()
        if head is None:
            head = YOLOXHead(80)

        self.backbone = backbone
        self.head = head

        self.init_yolo()

    def init_yolo(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03
        self.head.initialize_biases(prior_prob=0.01)

    # def forward(self, x, targets=None):
    #     # fpn output content features of [dark3, dark4, dark5]
    #     fpn_outs = self.backbone(x)

    #     if self.training:
    #         assert targets is not None
    #         outputs, loss_dict = self.head(fpn_outs, targets, x)
    #         return outputs, loss_dict
    #     else:
    #         outputs = self.head(fpn_outs)
    #         return outputs

    def forward(self, x, targets=None, augment=False, cfg=None):
        if augment:
            assert not self.training, "multiscale training is not implemented"
            img_size = x.shape[-2:]
            scales = cfg.scales
            # flips = [None, 3, None]  # flips (2-ud, 3-lr)
            det_preds = []
            # for si, fi in zip(scales, flips):
            for si in scales:
                # xi = scale_img(x.flip(fi) if fi else x, si, gs=32)
                xi = scale_img(x, si, gs=32)
                yi = self.forward_once(xi, targets)
                yi["det_preds"][:, :, :4] /= si  # de-scale
                # if fi == 2:
                #     yi["det_preds"][:, :, 1] = img_size[0] - yi["det_preds"][:, :, 1]  # de-flip ud
                # elif fi == 3:
                #     yi["det_preds"][:, :, 0] = img_size[1] - yi["det_preds"][:, :, 0]  # de-flip lr
                # adaptive small medium large objects
                # if si < 1:
                #     yi["det_preds"][:, :, 5] = torch.where(
                #         yi["det_preds"][:, :, 2] * yi["det_preds"][:, :, 3] < 96 * 96,
                #         yi["det_preds"][:, :, 5] * 0.6,
                #         yi["det_preds"][:, :, 5]
                #     )
                # elif si > 1:
                #     yi["det_preds"][:, :, 5] = torch.where(
                #         yi["det_preds"][:, :, 2] * yi["det_preds"][:, :, 3] > 32 * 32,
                #         yi["det_preds"][:, :, 5] * 0.6,
                #         yi["det_preds"][:, :, 5]
                #     )
                det_preds.append(yi["det_preds"])
            det_preds = torch.cat(det_preds, 1)
            outputs = dict(det_preds=det_preds)
            return outputs  # augmented inference, train #TODO multi-scale train
        else:
            return self.forward_once(x, targets)

    def forward_once(self, x, targets=None):
        # fpn output content features of [dark3, dark4, dark5]
        fpn_outs = self.backbone(x)

        if self.training:
            assert targets is not None
            outputs, loss_dict = self.head(fpn_outs, targets, x)
            return outputs, loss_dict
        else:
            outputs = self.head(fpn_outs)
            return outputs
