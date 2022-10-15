#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import random

import cv2
import numpy as np
from core.utils.my_comm import get_local_rank
from core.utils.augment import AugmentRGB

from det.yolox.utils import adjust_box_anns

from ..data_augment import random_affine, augment_hsv
from .datasets_wrapper import Dataset


def get_mosaic_coordinate(mosaic_image, mosaic_index, xc, yc, w, h, input_h, input_w):
    # TODO update doc
    # index0 to top left part of image
    if mosaic_index == 0:
        x1, y1, x2, y2 = max(xc - w, 0), max(yc - h, 0), xc, yc
        small_coord = w - (x2 - x1), h - (y2 - y1), w, h
    # index1 to top right part of image
    elif mosaic_index == 1:
        x1, y1, x2, y2 = xc, max(yc - h, 0), min(xc + w, input_w * 2), yc
        small_coord = 0, h - (y2 - y1), min(w, x2 - x1), h
    # index2 to bottom left part of image
    elif mosaic_index == 2:
        x1, y1, x2, y2 = max(xc - w, 0), yc, xc, min(input_h * 2, yc + h)
        small_coord = w - (x2 - x1), 0, w, min(y2 - y1, h)
    # index2 to bottom right part of image
    elif mosaic_index == 3:
        x1, y1, x2, y2 = (
            xc,
            yc,
            min(xc + w, input_w * 2),
            min(input_h * 2, yc + h),
        )  # noqa
        small_coord = 0, 0, min(w, x2 - x1), min(y2 - y1, h)
    return (x1, y1, x2, y2), small_coord


class MosaicDetection(Dataset):
    """Detection dataset wrapper that performs mixup for normal dataset."""

    def __init__(
        self,
        img_size,
        mosaic=True,
        preproc=None,
        degrees=10.0,
        translate=0.1,
        mosaic_scale=(0.5, 1.5),
        mixup_scale=(0.5, 1.5),
        shear=2.0,
        enable_mixup=True,
        mosaic_prob=1.0,
        mixup_prob=1.0,
        COLOR_AUG_PROB=0.0,
        COLOR_AUG_TYPE="",
        COLOR_AUG_CODE=(),
        AUG_HSV_PROB=0,
        HSV_H=0,
        HSV_S=0,
        HSV_V=0,
        FORMAT="RGB",
        *args
    ):
        """

        Args:
            img_size (tuple): (h, w)
            mosaic (bool): enable mosaic augmentation or not.
            preproc (func):
            degrees (float):
            translate (float):
            mosaic_scale (tuple):
            mixup_scale (tuple):
            shear (float):
            enable_mixup (bool):
            *args(tuple) : Additional arguments for mixup random sampler.
        """
        super().__init__(img_size, mosaic=mosaic)
        self.preproc = preproc
        self.degrees = degrees
        self.translate = translate
        self.scale = mosaic_scale
        self.shear = shear
        self.mixup_scale = mixup_scale
        self.enable_mosaic = mosaic
        self.enable_mixup = enable_mixup
        self.mosaic_prob = mosaic_prob
        self.mixup_prob = mixup_prob
        self.local_rank = get_local_rank()

        # color aug config
        self.color_aug_prob = COLOR_AUG_PROB
        self.color_aug_type = COLOR_AUG_TYPE
        self.color_aug_code = COLOR_AUG_CODE

        # hsv aug config
        self.aug_hsv_prob = AUG_HSV_PROB
        self.hsv_h = HSV_H
        self.hsv_s = HSV_S
        self.hsv_v = HSV_V
        self.img_format = FORMAT

        if self.color_aug_prob > 0:
            self.color_augmentor = self._get_color_augmentor(aug_type=self.color_aug_type, aug_code=self.color_aug_code)
        else:
            self.color_augmentor = None

    def init_dataset(self, dataset):
        # dataset(Dataset) : Pytorch dataset object.
        self._dataset = dataset
        return self

    def __len__(self):
        return len(self._dataset)

    @Dataset.mosaic_getitem
    def __getitem__(self, idx):
        if self.enable_mosaic and random.random() < self.mosaic_prob:
            mosaic_labels = []
            input_dim = self._dataset.input_dim
            input_h, input_w = input_dim[0], input_dim[1]

            # yc, xc = s, s  # mosaic center x, y
            yc = int(random.uniform(0.5 * input_h, 1.5 * input_h))
            xc = int(random.uniform(0.5 * input_w, 1.5 * input_w))

            # 3 additional image indices
            indices = [idx] + [random.randint(0, len(self._dataset) - 1) for _ in range(3)]

            for i_mosaic, index in enumerate(indices):
                img, _labels, scene_im_id, _, img_id = self._dataset.pull_item(index)
                h0, w0 = img.shape[:2]  # orig hw
                scale = min(1.0 * input_h / h0, 1.0 * input_w / w0)
                img = cv2.resize(
                    img,
                    (int(w0 * scale), int(h0 * scale)),
                    interpolation=cv2.INTER_LINEAR,
                )
                # generate output mosaic image
                (h, w, c) = img.shape[:3]
                if i_mosaic == 0:
                    mosaic_img = np.full((input_h * 2, input_w * 2, c), 114, dtype=np.uint8)

                # suffix l means large image, while s means small image in mosaic aug.
                (l_x1, l_y1, l_x2, l_y2), (
                    s_x1,
                    s_y1,
                    s_x2,
                    s_y2,
                ) = get_mosaic_coordinate(mosaic_img, i_mosaic, xc, yc, w, h, input_h, input_w)

                mosaic_img[l_y1:l_y2, l_x1:l_x2] = img[s_y1:s_y2, s_x1:s_x2]
                padw, padh = l_x1 - s_x1, l_y1 - s_y1

                labels = _labels.copy()
                # Normalized xywh to pixel xyxy format
                if _labels.size > 0:
                    labels[:, 0] = scale * _labels[:, 0] + padw
                    labels[:, 1] = scale * _labels[:, 1] + padh
                    labels[:, 2] = scale * _labels[:, 2] + padw
                    labels[:, 3] = scale * _labels[:, 3] + padh
                mosaic_labels.append(labels)

            if len(mosaic_labels):
                mosaic_labels = np.concatenate(mosaic_labels, 0)
                np.clip(mosaic_labels[:, 0], 0, 2 * input_w, out=mosaic_labels[:, 0])
                np.clip(mosaic_labels[:, 1], 0, 2 * input_h, out=mosaic_labels[:, 1])
                np.clip(mosaic_labels[:, 2], 0, 2 * input_w, out=mosaic_labels[:, 2])
                np.clip(mosaic_labels[:, 3], 0, 2 * input_h, out=mosaic_labels[:, 3])

            mosaic_img, mosaic_labels = random_affine(
                mosaic_img,
                mosaic_labels,
                target_size=(input_w, input_h),
                degrees=self.degrees,
                translate=self.translate,
                scales=self.scale,
                shear=self.shear,
            )

            # -----------------------------------------------------------------
            # CopyPaste: https://arxiv.org/abs/2012.07177
            # -----------------------------------------------------------------
            if self.enable_mixup and not len(mosaic_labels) == 0 and random.random() < self.mixup_prob:
                mosaic_img, mosaic_labels = self.mixup(mosaic_img, mosaic_labels, self.input_dim)
            mix_img, padded_labels = self.preproc(mosaic_img, mosaic_labels, self.input_dim)
            img_info = (mix_img.shape[1], mix_img.shape[0])

            # Augment colorspace
            dtype = mix_img.dtype
            mix_img = mix_img.transpose(2, 1, 0).astype(np.uint8).copy()
            # cv2.imwrite(f'output/transposed.png', mix_img)
            if np.random.rand() < self.aug_hsv_prob:
                augment_hsv(
                    mix_img,
                    hgain=self.hsv_h,
                    sgain=self.hsv_s,
                    vgain=self.hsv_v,
                    source_format=self.img_format,
                )

            # color augment
            if self.color_aug_prob > 0 and self.color_augmentor is not None:
                if np.random.rand() < self.color_aug_prob:
                    mix_img = self._color_aug(mix_img, self.color_aug_type)
            mix_img = mix_img.transpose(2, 1, 0).astype(dtype).copy()
            # cv2.imwrite('output/transposed_back.png', mix_img.astype(dtype).copy())

            # -----------------------------------------------------------------
            # img_info and img_id are not used for training.
            # They are also hard to be specified on a mosaic image.
            # -----------------------------------------------------------------
            return mix_img, padded_labels, scene_im_id, img_info, img_id

        else:
            self._dataset._input_dim = self.input_dim
            img, label, scene_im_id, img_info, img_id = self._dataset.pull_item(idx)
            img, label = self.preproc(img, label, self.input_dim)
            return img, label, scene_im_id, img_info, img_id

    def mixup(self, origin_img, origin_labels, input_dim):
        jit_factor = random.uniform(*self.mixup_scale)
        FLIP = random.uniform(0, 1) > 0.5
        cp_labels = []
        while len(cp_labels) == 0:
            cp_index = random.randint(0, self.__len__() - 1)
            cp_labels = self._dataset.load_anno(cp_index)
        img, cp_labels, _, _, _ = self._dataset.pull_item(cp_index)

        if len(img.shape) == 3:
            cp_img = np.ones((input_dim[0], input_dim[1], 3), dtype=np.uint8) * 114
        else:
            cp_img = np.ones(input_dim, dtype=np.uint8) * 114
        cp_scale_ratio = min(input_dim[0] / img.shape[0], input_dim[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * cp_scale_ratio), int(img.shape[0] * cp_scale_ratio)),
            interpolation=cv2.INTER_LINEAR,
        )

        cp_img[: int(img.shape[0] * cp_scale_ratio), : int(img.shape[1] * cp_scale_ratio)] = resized_img

        cp_img = cv2.resize(
            cp_img,
            (int(cp_img.shape[1] * jit_factor), int(cp_img.shape[0] * jit_factor)),
        )
        cp_scale_ratio *= jit_factor

        if FLIP:
            cp_img = cp_img[:, ::-1, :]

        origin_h, origin_w = cp_img.shape[:2]
        target_h, target_w = origin_img.shape[:2]
        padded_img = np.zeros((max(origin_h, target_h), max(origin_w, target_w), 3), dtype=np.uint8)
        padded_img[:origin_h, :origin_w] = cp_img

        x_offset, y_offset = 0, 0
        if padded_img.shape[0] > target_h:
            y_offset = random.randint(0, padded_img.shape[0] - target_h - 1)
        if padded_img.shape[1] > target_w:
            x_offset = random.randint(0, padded_img.shape[1] - target_w - 1)
        padded_cropped_img = padded_img[y_offset : y_offset + target_h, x_offset : x_offset + target_w]

        cp_bboxes_origin_np = adjust_box_anns(cp_labels[:, :4].copy(), cp_scale_ratio, 0, 0, origin_w, origin_h)
        if FLIP:
            cp_bboxes_origin_np[:, 0::2] = origin_w - cp_bboxes_origin_np[:, 0::2][:, ::-1]
        cp_bboxes_transformed_np = cp_bboxes_origin_np.copy()
        cp_bboxes_transformed_np[:, 0::2] = np.clip(cp_bboxes_transformed_np[:, 0::2] - x_offset, 0, target_w)
        cp_bboxes_transformed_np[:, 1::2] = np.clip(cp_bboxes_transformed_np[:, 1::2] - y_offset, 0, target_h)

        cls_labels = cp_labels[:, 4:5].copy()
        box_labels = cp_bboxes_transformed_np
        labels = np.hstack((box_labels, cls_labels))
        origin_labels = np.vstack((origin_labels, labels))
        origin_img = origin_img.astype(np.float32)
        origin_img = 0.5 * origin_img + 0.5 * padded_cropped_img.astype(np.float32)

        return origin_img.astype(np.uint8), origin_labels

    def _get_color_augmentor(self, aug_type="ROI10D", aug_code=None):
        # fmt: off
        if aug_type.lower() == "roi10d":
            color_augmentor = AugmentRGB(
                brightness_delta=2.5 / 255.,  # 0,
                lighting_std=0.3,
                saturation_var=(0.95, 1.05),  #(1, 1),
                contrast_var=(0.95, 1.05))  # (1, 1))  #
        elif aug_type.lower() == "aae":
            import imgaug.augmenters as iaa  # noqa
            from imgaug.augmenters import (Sequential, SomeOf, OneOf, Sometimes, WithColorspace, WithChannels, Noop,
                                           Lambda, AssertLambda, AssertShape, Scale, CropAndPad, Pad, Crop, Fliplr,
                                           Flipud, Superpixels, ChangeColorspace, PerspectiveTransform, Grayscale,
                                           GaussianBlur, AverageBlur, MedianBlur, Convolve, Sharpen, Emboss, EdgeDetect,
                                           DirectedEdgeDetect, Add, AddElementwise, AdditiveGaussianNoise, Multiply,
                                           MultiplyElementwise, Dropout, CoarseDropout, Invert, ContrastNormalization,
                                           Affine, PiecewiseAffine, ElasticTransformation, pillike, LinearContrast)  # noqa
            aug_code = """Sequential([
                # Sometimes(0.5, PerspectiveTransform(0.05)),
                # Sometimes(0.5, CropAndPad(percent=(-0.05, 0.1))),
                # Sometimes(0.5, Affine(scale=(1.0, 1.2))),
                Sometimes(0.5, CoarseDropout( p=0.2, size_percent=0.05) ),
                Sometimes(0.5, GaussianBlur(1.2*np.random.rand())),
                Sometimes(0.5, Add((-25, 25), per_channel=0.3)),
                Sometimes(0.3, Invert(0.2, per_channel=True)),
                Sometimes(0.5, Multiply((0.6, 1.4), per_channel=0.5)),
                Sometimes(0.5, Multiply((0.6, 1.4))),
                Sometimes(0.5, LinearContrast((0.5, 2.2), per_channel=0.3))
                ], random_order = False)"""
            # for darker objects, e.g. LM driller: use BOOTSTRAP_RATIO: 16 and weaker augmentation
            aug_code_weaker = """Sequential([
                Sometimes(0.4, CoarseDropout( p=0.1, size_percent=0.05) ),
                # Sometimes(0.5, Affine(scale=(1.0, 1.2))),
                Sometimes(0.5, GaussianBlur(np.random.rand())),
                Sometimes(0.5, Add((-20, 20), per_channel=0.3)),
                Sometimes(0.4, Invert(0.20, per_channel=True)),
                Sometimes(0.5, Multiply((0.7, 1.4), per_channel=0.8)),
                Sometimes(0.5, Multiply((0.7, 1.4))),
                Sometimes(0.5, LinearContrast((0.5, 2.0), per_channel=0.3))
                ], random_order=False)"""
            color_augmentor = eval(aug_code)
        elif aug_type.lower() == "code":  # assume imgaug
            import imgaug.augmenters as iaa
            from imgaug.augmenters import (Sequential, SomeOf, OneOf, Sometimes, WithColorspace, WithChannels, Noop,
                                           Lambda, AssertLambda, AssertShape, Scale, CropAndPad, Pad, Crop, Fliplr,
                                           Flipud, Superpixels, ChangeColorspace, PerspectiveTransform, Grayscale,
                                           GaussianBlur, AverageBlur, MedianBlur, Convolve, Sharpen, Emboss, EdgeDetect,
                                           DirectedEdgeDetect, Add, AddElementwise, AdditiveGaussianNoise, Multiply,
                                           MultiplyElementwise, Dropout, CoarseDropout, Invert, ContrastNormalization,
                                           Affine, PiecewiseAffine, ElasticTransformation, pillike, LinearContrast, Canny)  # noqa
            aug_code = self.color_aug_code
            color_augmentor = eval(aug_code)
        elif aug_type.lower() == 'code_albu':
            from albumentations import (HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
                                        Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion,
                                        HueSaturationValue, IAAAdditiveGaussianNoise, GaussNoise, MotionBlur,
                                        MedianBlur, IAAPiecewiseAffine, IAASharpen, IAAEmboss, RandomContrast,
                                        RandomBrightness, Flip, OneOf, Compose, CoarseDropout, RGBShift, RandomGamma,
                                        RandomBrightnessContrast, JpegCompression, InvertImg)  # noqa
            aug_code = """Compose([
                CoarseDropout(max_height=0.05*480, max_holes=0.05*640, p=0.4),
                OneOf([
                    IAAAdditiveGaussianNoise(p=0.5),
                    GaussNoise(p=0.5),
                ], p=0.2),
                OneOf([
                    MotionBlur(p=0.2),
                    MedianBlur(blur_limit=3, p=0.1),
                    Blur(blur_limit=3, p=0.1),
                ], p=0.2),
                OneOf([
                    CLAHE(clip_limit=2),
                    IAASharpen(),
                    IAAEmboss(),
                    RandomBrightnessContrast(),
                ], p=0.3),
                InvertImg(p=0.2),
                RGBShift(r_shift_limit=105, g_shift_limit=45, b_shift_limit=40, p=0.5),
                RandomContrast(limit=0.9, p=0.5),
                RandomGamma(gamma_limit=(80,120), p=0.5),
                RandomBrightness(limit=1.2, p=0.5),
                HueSaturationValue(hue_shift_limit=172, sat_shift_limit=20, val_shift_limit=27, p=0.3),
                JpegCompression(quality_lower=4, quality_upper=100, p=0.4),
            ], p=0.8)"""
            color_augmentor = eval(self.color_aug_code)
        else:
            color_augmentor = None
        # fmt: on
        return color_augmentor

    def _color_aug(self, image, aug_type="ROI10D"):
        # assume image in [0, 255] uint8
        if aug_type.lower() == "roi10d":  # need normalized image in [0,1]
            image = np.asarray(image / 255.0, dtype=np.float32).copy()
            image = self.color_augmentor.augment(image)
            image = (image * 255.0 + 0.5).astype(np.uint8)
            return image
        elif aug_type.lower() in ["aae", "code"]:
            # imgaug need uint8
            return self.color_augmentor.augment_image(image)
        elif aug_type.lower() in ["code_albu"]:
            augmented = self.color_augmentor(image=image)
            return augmented["image"]
        else:
            raise ValueError("aug_type: {} is not supported.".format(aug_type))
