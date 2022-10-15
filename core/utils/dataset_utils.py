import copy
import logging
import numpy as np
import operator
import pickle
import random
import mmcv
import torch
import torch.multiprocessing as mp
import torch.utils.data as data
from torch.utils.data import dataloader

from detectron2.utils.serialize import PicklableWrapper
from detectron2.data.build import worker_init_reset_seed, get_detection_dataset_dicts
from detectron2.data.common import AspectRatioGroupedDataset, DatasetFromList, MapDataset
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.samplers import InferenceSampler, RepeatFactorTrainingSampler, TrainingSampler
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
import ref
from . import my_comm as comm
from lib.utils.setup_logger import log_first_n


logger = logging.getLogger(__name__)


def flat_dataset_dicts(dataset_dicts):
    """
    flatten the dataset dicts of detectron2 format
    original: list of dicts, each dict contains some image-level infos
              and an "annotations" field for instance-level infos of multiple instances
    => flat the instance level annotations
    flat format:
        list of dicts,
            each dict includes the image/instance-level infos
            an `inst_id` of a single instance,
            `inst_infos` includes only one instance
    """
    new_dicts = []
    for dataset_dict in dataset_dicts:
        img_infos = {_k: _v for _k, _v in dataset_dict.items() if _k not in ["annotations"]}
        if "annotations" in dataset_dict:
            for inst_id, anno in enumerate(dataset_dict["annotations"]):
                rec = {"inst_id": inst_id, "inst_infos": anno}
                rec.update(img_infos)
                new_dicts.append(rec)
        else:
            rec = img_infos
            new_dicts.append(rec)
    return new_dicts


def remove_anno_keys_dataset_dicts(dataset_dicts, keys=[]):
    """remove the keys in annotations."""
    new_dicts = []
    for dataset_dict in dataset_dicts:
        new_d = {_k: _v for _k, _v in dataset_dict.items() if _k not in ["annotations"]}
        if "annotations" not in dataset_dict:
            new_dicts.append(new_d)
            continue

        new_annos = []
        for inst_id, anno in enumerate(dataset_dict["annotations"]):
            new_anno = {_k: _v for _k, _v in anno.items() if _k not in keys}
            new_annos.append(new_anno)

            removed_keys = []
            for _k in keys:
                if _k in anno and _k not in new_anno:
                    removed_keys.append(_k)
            log_first_n(logging.WARNING, f"removed keys: {removed_keys} from annotations", n=1)

        if len(new_annos) > 0:
            new_d["annotations"] = new_annos
        new_dicts.append(new_d)
    return new_dicts


def filter_invalid_in_dataset_dicts(dataset_dicts, visib_thr=0.0):
    """
    filter invalid instances in the dataset_dicts (for train)
    Args:
        visib_thr:
    """
    num_filtered = 0
    new_dicts = []
    for dataset_dict in dataset_dicts:
        new_dict = {_k: _v for _k, _v in dataset_dict.items() if _k not in ["annotations"]}
        if "annotations" in dataset_dict:
            new_annos = []
            for inst_id, anno in enumerate(dataset_dict["annotations"]):
                if anno.get("visib_fract", 1.0) > visib_thr:
                    new_annos.append(anno)
                else:
                    num_filtered += 1
            if len(new_annos) == 0:
                continue
            new_dict["annotations"] = new_annos

        new_dicts.append(new_dict)
    if num_filtered > 0:
        logger.warning(f"filtered out {num_filtered} instances with visib_fract <= {visib_thr}")
    return new_dicts


def trivial_batch_collator(batch):
    """A batch collator that does nothing.

    https://github.com/pytorch/fairseq/issues/1171
    """
    dataloader._use_shared_memory = False
    return batch


def filter_empty_dets(dataset_dicts):
    """
    Filter out images with empty detections
    NOTE: here we assume detections are in "annotations"
    Args:
        dataset_dicts (list[dict]): annotations in Detectron2 Dataset format.

    Returns:
        list[dict]: the same format, but filtered.
    """
    num_before = len(dataset_dicts)

    def valid(anns):
        if len(anns) > 0:
            return True
        # for ann in anns:
        #     if ann.get("iscrowd", 0) == 0:
        #         return True
        return False

    dataset_dicts = [x for x in dataset_dicts if valid(x["annotations"])]
    num_after = len(dataset_dicts)
    if num_after < num_before:
        logger = logging.getLogger(__name__)
        logger.warning(
            "Removed {} images with empty detections. {} images left.".format(num_before - num_after, num_after)
        )
    return dataset_dicts


def load_detections_into_dataset(
    dataset_name,
    dataset_dicts,
    det_file,
    top_k_per_obj=1,
    score_thr=0.0,
    train_objs=None,
    selected_scenes=None,
):
    """Load test detections into the dataset.

    Args:
        dataset_name (str):
        dataset_dicts (list[dict]): annotations in Detectron2 Dataset format.
        det_file (str): file path of pre-computed detections, in json format.

    Returns:
        list[dict]: the same format as dataset_dicts, but added proposal field.
    """

    logger.info("Loading detections for {} from: {}".format(dataset_name, det_file))
    detections = mmcv.load(det_file)

    meta = MetadataCatalog.get(dataset_name)
    objs = meta.objs
    ref_key = meta.ref_key
    data_ref = ref.__dict__[ref_key]
    models_info = data_ref.get_models_info()

    if "annotations" in dataset_dicts[0]:
        logger.warning("pop the original annotations, load detections")
    new_dataset_dicts = []
    for i, record_ori in enumerate(dataset_dicts):
        record = copy.deepcopy(record_ori)
        scene_im_id = record["scene_im_id"]
        if scene_im_id not in detections:  # not detected
            logger.warning(f"no detections found in {scene_im_id}")
            continue

        scene_id = int(scene_im_id.split("/")[0])
        if selected_scenes is not None:
            if isinstance(selected_scenes, int):
                selected_scenes = [selected_scenes]
            if scene_id not in selected_scenes:
                continue
        dets_i = detections[scene_im_id]

        annotations = []
        obj_annotations = {obj: [] for obj in objs}
        for det in dets_i:
            obj_id = det["obj_id"]
            bbox_est = det["bbox_est"]  # xywh
            time = det.get("time", 0.0)
            score = det.get("score", 1.0)
            if score < score_thr:
                continue
            obj_name = data_ref.id2obj[obj_id]

            if obj_name not in objs:  # detected obj is not interested
                continue

            if train_objs is not None:  # not in trained objects
                if obj_name not in train_objs:
                    continue

            label = objs.index(obj_name)
            inst = {
                "category_id": label,
                "bbox_est": bbox_est,
                "bbox_mode": BoxMode.XYWH_ABS,
                "score": score,
                "time": time,
                "model_info": models_info[str(obj_id)],  # TODO: maybe just load this in the main function
            }
            obj_annotations[obj_name].append(inst)
        for obj, cur_annos in obj_annotations.items():
            scores = [ann["score"] for ann in cur_annos]
            sel_annos = [ann for _, ann in sorted(zip(scores, cur_annos), key=lambda pair: pair[0], reverse=True)][
                :top_k_per_obj
            ]
            annotations.extend(sel_annos)
        # NOTE: maybe [], no detections
        record["annotations"] = annotations
        new_dataset_dicts.append(record)

    if len(new_dataset_dicts) < len(dataset_dicts):
        logger.warning(
            "No detections found in {} images. original: {} imgs, left: {} imgs".format(
                len(dataset_dicts) - len(new_dataset_dicts),
                len(dataset_dicts),
                len(new_dataset_dicts),
            )
        )
    return new_dataset_dicts


def load_init_poses_into_dataset(
    dataset_name,
    dataset_dicts,
    init_pose_file,
    top_k_per_obj=1,
    score_thr=0.0,
    train_objs=None,
    selected_scenes=None,
):
    """Load initial poses into the dataset.

    Args:
        dataset_name (str):
        dataset_dicts (list[dict]): annotations in Detectron2 Dataset format.
        init_pose_file (str): file path of pre-computed initial poses, in json format.

    Returns:
        list[dict]: the same format as dataset_dicts, but added proposal field.
    """

    logger.info("Loading initial poses for {} from: {}".format(dataset_name, init_pose_file))
    init_det_poses = mmcv.load(init_pose_file)

    meta = MetadataCatalog.get(dataset_name)
    objs = meta.objs
    ref_key = meta.ref_key
    data_ref = ref.__dict__[ref_key]
    models_info = data_ref.get_models_info()

    if "annotations" in dataset_dicts[0]:
        logger.warning("pop the original annotations, load initial poses")
    for record in dataset_dicts:
        scene_im_id = record["scene_im_id"]
        if scene_im_id not in init_det_poses:  # not detected
            logger.warning(f"no init pose detections found in {scene_im_id}")
            record["annotations"] = []  # NOTE: no init pose, should be ignored
            continue

        scene_id = int(scene_im_id.split("/")[0])
        if selected_scenes is not None:
            if isinstance(selected_scenes, int):
                selected_scenes = [selected_scenes]
            if scene_id not in selected_scenes:
                continue

        dets_i = init_det_poses[scene_im_id]

        annotations = []
        obj_annotations = {obj: [] for obj in objs}
        for det in dets_i:
            obj_id = det["obj_id"]
            # NOTE: need to prepare init poses into this format
            pose_est = np.array(det["pose_est"], dtype=np.float32).reshape(3, 4)
            bbox_est = det.get("bbox_est", None)  # xywh or None
            time = det.get("time", 0.0)
            score = det.get("score", 1.0)
            if score < score_thr:
                continue
            obj_name = data_ref.id2obj[obj_id]

            if obj_name not in objs:  # detected obj is not interested
                continue

            if train_objs is not None:  # not in trained objects
                if obj_name not in train_objs:
                    continue

            label = objs.index(obj_name)
            inst = {
                "category_id": label,
                "pose_est": pose_est,
                "score": score,
                "time": time,
                "model_info": models_info[str(obj_id)],  # TODO: maybe just load this in the main function
            }
            if bbox_est is not None:  # if None, compute bboxes from poses and 3D points later
                inst["bbox_est"] = bbox_est
                inst["bbox_mode"] = BoxMode.XYWH_ABS
            obj_annotations[obj_name].append(inst)
        for obj, cur_annos in obj_annotations.items():
            scores = [ann["score"] for ann in cur_annos]
            sel_annos = [ann for _, ann in sorted(zip(scores, cur_annos), key=lambda pair: pair[0], reverse=True)][
                :top_k_per_obj
            ]
            annotations.extend(sel_annos)
        # NOTE: maybe [], no initial poses
        record["annotations"] = annotations

    return dataset_dicts


def load_catre_init_into_dataset(
    dataset_name,
    dataset_dicts,
    init_pose_file,
    score_thr=0.0,
    train_objs=None,
    with_masks=True,
    with_bboxes=True,
):
    """Load initial poses into the dataset.

    Args:
        dataset_name (str):
        dataset_dicts (list[dict]): annotations in Detectron2 Dataset format.
        init_pose_file (str): file path of pre-computed initial poses, in json format.

    Returns:
        list[dict]: the same format as dataset_dicts, but added proposal field.
    """

    logger.info("Loading initial poses for {} from: {}".format(dataset_name, init_pose_file))
    init_det_poses = mmcv.load(init_pose_file)

    meta = MetadataCatalog.get(dataset_name)
    objs = meta.objs
    ref_key = meta.ref_key
    data_ref = ref.__dict__[ref_key]

    if "annotations" in dataset_dicts[0]:
        logger.warning("pop the original annotations, load initial poses")
    for record in dataset_dicts:
        scene_im_id = record["scene_im_id"]
        if scene_im_id not in init_det_poses:  # not detected
            logger.warning(f"no init pose detections found in {scene_im_id}")
            record["annotations"] = []  # NOTE: no init pose, should be ignored
            continue
        dets_i = init_det_poses[scene_im_id]

        annotations = []
        obj_annotations = {obj: [] for obj in objs}
        for det in dets_i:
            obj_id = det["obj_id"]
            # NOTE: need to prepare init poses into this format
            pose_est = np.array(det["pose_est"], dtype=np.float32).reshape(3, 4)
            scale_est = np.array(det["scale_est"], dtype=np.float32)  # shape(3,)
            time = det.get("time", 0.0)
            score = det.get("score", 1.0)
            mug_handle = det.get("mug_handle", 1)
            if score < score_thr:
                continue
            obj_name = data_ref.id2obj[obj_id]

            if obj_name not in objs:  # detected obj is not interested
                continue

            if train_objs is not None:  # not in trained objects
                if obj_name not in train_objs:
                    continue

            label = objs.index(obj_name)
            inst = {
                "category_id": label,  # 0-based label
                "pose_est": pose_est,
                "scale_est": scale_est,
                "obj_name": obj_name,
                "mug_handle": mug_handle,
                "score": score,
                "time": time,
            }
            if with_bboxes:
                inst["bbox_est"] = det["bbox_est"]
                inst["bbox_mode"] = BoxMode.XYXY_ABS
            if with_masks:
                inst["segmentation"] = det["segmentation"]  # overwrite gt masks
            obj_annotations[obj_name].append(inst)
        for obj, cur_annos in obj_annotations.items():
            scores = [ann["score"] for ann in cur_annos]
            sel_annos = [ann for _, ann in sorted(zip(scores, cur_annos), key=lambda pair: pair[0], reverse=True)]
            annotations.extend(sel_annos)
        # NOTE: maybe [], no initial poses
        record["annotations"] = annotations

    return dataset_dicts


def my_build_batch_data_loader(
    dataset,
    sampler,
    total_batch_size,
    *,
    aspect_ratio_grouping=False,
    num_workers=0,
    persistent_workers=False,
):
    """Build a batched dataloader for training.

    Args:
        dataset (torch.utils.data.Dataset): map-style PyTorch dataset. Can be indexed.
        sampler (torch.utils.data.sampler.Sampler): a sampler that produces indices
        total_batch_size, aspect_ratio_grouping, num_workers): see
            :func:`build_detection_train_loader`.
    Returns:
        iterable[list]. Length of each list is the batch size of the current
            GPU. Each element in the list comes from the dataset.
    """
    world_size = comm.get_world_size()
    assert (
        total_batch_size > 0 and total_batch_size % world_size == 0
    ), "Total batch size ({}) must be divisible by the number of gpus ({}).".format(total_batch_size, world_size)

    batch_size = total_batch_size // world_size

    # Horovod: limit # of CPU threads to be used per worker.
    # if num_workers > 0:
    #     torch.set_num_threads(num_workers)

    kwargs = {"num_workers": num_workers}
    # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to prevent
    # issues with Infiniband implementations that are not fork-safe
    # https://github.com/horovod/horovod/blob/master/examples/pytorch/pytorch_imagenet_resnet50.py
    # if (num_workers > 0 and hasattr(mp, '_supports_context') and
    #         mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
    #     kwargs['multiprocessing_context'] = 'forkserver'

    if aspect_ratio_grouping:
        data_loader = torch.utils.data.DataLoader(
            dataset,
            sampler=sampler,
            batch_sampler=None,
            collate_fn=operator.itemgetter(0),  # don't batch, but yield individual elements
            worker_init_fn=worker_init_reset_seed,
            persistent_workers=persistent_workers,
            **kwargs,
        )  # yield individual mapped dict
        return AspectRatioGroupedDataset(data_loader, batch_size)
    else:
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, batch_size, drop_last=True
        )  # drop_last so the batch always have the same size
        return torch.utils.data.DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=trivial_batch_collator,
            worker_init_fn=worker_init_reset_seed,
            persistent_workers=persistent_workers,
            **kwargs,
        )
