#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import os
import random
import uuid

import numpy as np

import torch
from torch.utils.data.dataloader import DataLoader as torchDataLoader
from torch.utils.data.dataloader import default_collate
import operator

from detectron2.data.build import (
    AspectRatioGroupedDataset,
    worker_init_reset_seed,
    trivial_batch_collator,
    InferenceSampler,
)

from core.utils.my_comm import get_world_size

from .samplers import YoloBatchSampler, InfiniteSampler

# from .datasets import Base_DatasetFromList


class DataLoader(torchDataLoader):
    """Lightnet dataloader that enables on the fly resizing of the images.

    See :class:`torch.utils.data.DataLoader` for more information on the arguments.
    Check more on the following website:
    https://gitlab.com/EAVISE/lightnet/-/blob/master/lightnet/data/_dataloading.py
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__initialized = False
        shuffle = False
        batch_sampler = None
        if len(args) > 5:
            shuffle = args[2]
            sampler = args[3]
            batch_sampler = args[4]
        elif len(args) > 4:
            shuffle = args[2]
            sampler = args[3]
            if "batch_sampler" in kwargs:
                batch_sampler = kwargs["batch_sampler"]
        elif len(args) > 3:
            shuffle = args[2]
            if "sampler" in kwargs:
                sampler = kwargs["sampler"]
            if "batch_sampler" in kwargs:
                batch_sampler = kwargs["batch_sampler"]
        else:
            if "shuffle" in kwargs:
                shuffle = kwargs["shuffle"]
            if "sampler" in kwargs:
                sampler = kwargs["sampler"]
            if "batch_sampler" in kwargs:
                batch_sampler = kwargs["batch_sampler"]

        # Use custom BatchSampler
        if batch_sampler is None:
            if sampler is None:
                if shuffle:
                    sampler = torch.utils.data.sampler.RandomSampler(self.dataset)
                    # sampler = torch.utils.data.DistributedSampler(self.dataset)
                else:
                    sampler = torch.utils.data.sampler.SequentialSampler(self.dataset)
            batch_sampler = YoloBatchSampler(
                sampler,
                self.batch_size,
                self.drop_last,
                input_dimension=self.dataset.input_dim,
            )
            # batch_sampler = IterationBasedBatchSampler(batch_sampler, num_iterations =

        self.batch_sampler = batch_sampler

        self.__initialized = True

    def close_mosaic(self):
        self.batch_sampler.mosaic = False


# def list_collate(batch):
#     """
#     Function that collates lists or tuples together into one list (of lists/tuples).
#     Use this as the collate function in a Dataloader, if you want to have a list of
#     items as an output, as opposed to tensors (eg. Brambox.boxes).
#     """
#     items = list(zip(*batch))

#     for i in range(len(items)):
#         if isinstance(items[i][0], (list, tuple)):
#             items[i] = list(items[i])
#         else:
#             items[i] = default_collate(items[i])

#     return items


def build_yolox_batch_data_loader(
    dataset, sampler, total_batch_size, *, aspect_ratio_grouping=False, num_workers=0, pin_memory=False
):
    """
    Build a batched dataloader. The main differences from `torch.utils.data.DataLoader` are:
    1. support aspect ratio grouping options
    2. use no "batch collation", because this is common for detection training

    Args:
        dataset (torch.utils.data.Dataset): map-style PyTorch dataset. Can be indexed.
        sampler (torch.utils.data.sampler.Sampler): a sampler that produces indices
        total_batch_size, aspect_ratio_grouping, num_workers): see
            :func:`build_detection_train_loader`.

    Returns:
        iterable[list]. Length of each list is the batch size of the current
            GPU. Each element in the list comes from the dataset.
    """
    world_size = get_world_size()
    assert (
        total_batch_size > 0 and total_batch_size % world_size == 0
    ), "Total batch size ({}) must be divisible by the number of gpus ({}).".format(total_batch_size, world_size)

    batch_size = total_batch_size // world_size
    if aspect_ratio_grouping:
        data_loader = torch.utils.data.DataLoader(
            dataset,
            sampler=sampler,
            num_workers=num_workers,
            batch_sampler=None,
            collate_fn=operator.itemgetter(0),  # don't batch, but yield individual elements
            worker_init_fn=worker_init_reset_seed,
        )  # yield individual mapped dict
        return AspectRatioGroupedDataset(data_loader, batch_size)
    else:
        # batch_sampler = torch.utils.data.sampler.BatchSampler(
        #     sampler, batch_size, drop_last=True
        # )  # drop_last so the batch always have the same size
        if hasattr(dataset, "enable_mosaic"):
            mosaic = dataset.enable_mosaic
        else:
            mosaic = False
        batch_sampler = YoloBatchSampler(
            mosaic=mosaic,
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,  # NOTE: different to d2
            # input_dimension=dataset.input_dim,
        )
        return DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            # collate_fn=trivial_batch_collator,  # TODO: use this when item is changed to dict
            worker_init_fn=worker_init_reset_seed,
            pin_memory=pin_memory,
        )


def build_yolox_train_loader(
    dataset,
    *,
    aug_wrapper,
    total_batch_size,
    sampler=None,
    aspect_ratio_grouping=False,
    num_workers=0,
    pin_memory=False,
    seed=None
):
    """Build a dataloader for object detection with some default features. This
    interface is experimental.

    Args:
        dataset (torch.utils.data.Dataset): Base_DatasetFromList
        aug_wrapper (callable): MosaciDetection
        total_batch_size (int): total batch size across all workers. Batching
            simply puts data into a list.
        sampler (torch.utils.data.sampler.Sampler or None): a sampler that produces
            indices to be applied on ``dataset``. Default to :class:`TrainingSampler`,
            which coordinates an infinite random shuffle sequence across all workers.
        aspect_ratio_grouping (bool): whether to group images with similar
            aspect ratio for efficiency. When enabled, it requires each
            element in dataset be a dict with keys "width" and "height".
        num_workers (int): number of parallel data loading workers

    Returns:
        torch.utils.data.DataLoader:
            a dataloader. Each output from it is a ``list[mapped_element]`` of length
            ``total_batch_size / num_workers``, where ``mapped_element`` is produced
            by the ``mapper``.
    """

    if aug_wrapper is not None:
        # MosaicDetection (mosaic, mixup, other augs)
        dataset = aug_wrapper.init_dataset(dataset)

    if sampler is None:
        # sampler = TrainingSampler(len(dataset))
        sampler = InfiniteSampler(len(dataset), seed=0 if seed is None else seed)
    assert isinstance(sampler, torch.utils.data.sampler.Sampler)
    return build_yolox_batch_data_loader(
        dataset,
        sampler,
        total_batch_size,
        aspect_ratio_grouping=aspect_ratio_grouping,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def build_yolox_test_loader(
    dataset, *, aug_wrapper=None, total_batch_size=1, sampler=None, num_workers=0, pin_memory=False
):
    """Similar to `build_detection_train_loader`, but uses a batch size of 1,
    and :class:`InferenceSampler`. This sampler coordinates all workers to
    produce the exact set of all samples. This interface is experimental.

    Args:
        dataset (list or torch.utils.data.Dataset): a list of dataset dicts,
            or a map-style pytorch dataset. They can be obtained by using
            :func:`DatasetCatalog.get` or :func:`get_detection_dataset_dicts`.
        aug_wrapper (callable): MosaciDetection
        total_batch_size (int): total batch size across all workers. Batching
            simply puts data into a list. Default test batch size is 1.
        sampler (torch.utils.data.sampler.Sampler or None): a sampler that produces
            indices to be applied on ``dataset``. Default to :class:`InferenceSampler`,
            which splits the dataset across all workers.
        num_workers (int): number of parallel data loading workers

    Returns:
        DataLoader: a torch DataLoader, that loads the given detection
        dataset, with test-time transformation and batching.

    Examples:
    ::
        data_loader = build_detection_test_loader(
            DatasetRegistry.get("my_test"),
            mapper=DatasetMapper(...))

        # or, instantiate with a CfgNode:
        data_loader = build_detection_test_loader(cfg, "my_test")
    """
    if aug_wrapper is not None:
        # MosaicDetection (mosaic, mixup, other augs)
        dataset = aug_wrapper.init_dataset(dataset)

    world_size = get_world_size()
    batch_size = total_batch_size // world_size
    if sampler is None:
        sampler = InferenceSampler(len(dataset))
    # Always use 1 image per worker during inference since this is the
    # standard when reporting inference time in papers.
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, batch_size, drop_last=False)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        # batch_size=batch_size,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        # collate_fn=trivial_batch_collator,
        pin_memory=pin_memory,
    )
    return data_loader


def yolox_worker_init_reset_seed(worker_id):
    seed = uuid.uuid4().int % 2**32
    random.seed(seed)
    torch.set_rng_state(torch.manual_seed(seed).get_state())
    np.random.seed(seed)
