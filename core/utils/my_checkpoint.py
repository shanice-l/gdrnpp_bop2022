import pickle
import os
from detectron2.utils.file_io import PathManager
from detectron2.checkpoint import DetectionCheckpointer
from mmcv.runner.checkpoint import (
    _load_checkpoint,
    load_state_dict,
    _process_mmcls_checkpoint,
)
import math
import logging
import timm
from timm.models.helpers import (
    load_state_dict_from_hf,
    load_state_dict_from_url,
    has_hf_hub,
    adapt_input_conv,
)
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from torch.nn.parallel import DataParallel, DistributedDataParallel
from pytorch_lightning.lite.wrappers import _LiteModule
from fairscale.nn.data_parallel.sharded_ddp import ShardedDataParallel


_logger = logging.getLogger(__name__)


class MyCheckpointer(DetectionCheckpointer):
    """https://github.com/aim-
    uofa/AdelaiDet/blob/master/adet/checkpoint/adet_checkpoint.py Same as
    :class:`DetectronCheckpointer`, but is able to convert models in AdelaiDet,
    such as LPF backbone."""

    def __init__(self, model, save_dir="", *, save_to_disk=None, prefix_to_remove=None, **checkpointables):
        # HACK: deal with lite model
        while isinstance(model, (DistributedDataParallel, DataParallel, _LiteModule, ShardedDataParallel)):
            model = model.module

        super().__init__(
            model,
            save_dir,
            save_to_disk=save_to_disk,
            **checkpointables,
        )
        self.prefix_to_remove = prefix_to_remove

    def _load_file(self, filename):
        if filename.endswith(".pkl"):
            with PathManager.open(filename, "rb") as f:
                data = pickle.load(f, encoding="latin1")
            if "model" in data and "__author__" in data:
                # file is in Detectron2 model zoo format
                self.logger.info("Reading a file from '{}'".format(data["__author__"]))
                return data
            else:
                # assume file is from Caffe2 / Detectron1 model zoo
                if "blobs" in data:
                    # Detection models have "blobs", but ImageNet models don't
                    data = data["blobs"]
                data = {k: v for k, v in data.items() if not k.endswith("_momentum")}
                if "weight_order" in data:
                    del data["weight_order"]
                return {
                    "model": data,
                    "__author__": "Caffe2",
                    "matching_heuristics": True,
                }

        if filename.startswith("torchvision://") or filename.startswith(("http://", "https://")):
            loaded = _load_checkpoint(filename)  # load torchvision pretrained model using mmcv
        else:
            loaded = super()._load_file(filename)  # load native pth checkpoint
        if "model" not in loaded:
            loaded = {"model": loaded}

        if self.prefix_to_remove is not None:
            # remove prefix such as "_module.", ".module"
            consume_prefix_in_state_dict_if_present(loaded["model"], self.prefix_to_remove)

        basename = os.path.basename(filename).lower()
        if "lpf" in basename or "dla" in basename:
            loaded["matching_heuristics"] = True
        return loaded


def load_mmcls_ckpt(model, filename, map_location=None, strict=False, logger=None):
    ckpt = _load_checkpoint(filename, map_location=map_location)
    # OrderedDict is a subclass of dict
    if not isinstance(ckpt, dict):
        raise RuntimeError(f"No state_dict found in checkpoint file {filename}")
    ckpt = _process_mmcls_checkpoint(ckpt)

    # get state_dict from checkpoint
    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt
    # strip prefix of state_dict
    if list(state_dict.keys())[0].startswith("module."):
        state_dict = {k[7:]: v for k, v in ckpt["state_dict"].items()}
    # load state_dict
    load_state_dict(model, state_dict, strict, logger)


def my_adapt_input_conv(in_chans, conv_weight, model_conv_weight):
    conv_type = model_conv_weight.dtype
    conv_weight = conv_weight.float()
    pretrained_ch = conv_weight.shape[1]
    res_conv_weight = model_conv_weight
    res_conv_weight[:, 0:pretrained_ch] = conv_weight
    res_conv_weight = res_conv_weight.to(conv_type)
    return res_conv_weight


def load_timm_pretrained(
    model,
    default_cfg=None,
    num_classes=1000,
    in_chans=3,
    filter_fn=None,
    strict=True,
    progress=True,
    adapt_input_mode="custom",
):
    """Load pretrained checkpoint
    Args:
        model (nn.Module) : PyTorch model module
        default_cfg (Optional[Dict]): default configuration for pretrained weights / target dataset
        num_classes (int): num_classes for model
        in_chans (int): in_chans for model
        filter_fn (Optional[Callable]): state_dict filter fn for load (takes state_dict, model as args)
        strict (bool): strict load of checkpoint
        progress (bool): enable progress bar for weight download
    """
    default_cfg = default_cfg or getattr(model, "default_cfg", None) or {}
    pretrained_url = default_cfg.get("url", None)
    hf_hub_id = default_cfg.get("hf_hub", None)
    if not pretrained_url and not hf_hub_id:
        _logger.warning("No pretrained weights exist for this model. Using random initialization.")
        return
    if hf_hub_id and has_hf_hub(necessary=not pretrained_url):
        _logger.info(f"Loading pretrained weights from Hugging Face hub ({hf_hub_id})")
        state_dict = load_state_dict_from_hf(hf_hub_id)
    else:
        _logger.info(f"Loading pretrained weights from url ({pretrained_url})")
        state_dict = load_state_dict_from_url(pretrained_url, progress=progress, map_location="cpu")
    if filter_fn is not None:
        # for backwards compat with filter fn that take one arg, try one first, the two
        try:
            state_dict = filter_fn(state_dict)
        except TypeError:
            state_dict = filter_fn(state_dict, model)

    input_convs = default_cfg.get("first_conv", None)
    if input_convs is not None and in_chans != 3:
        if isinstance(input_convs, str):
            input_convs = (input_convs,)
        for input_conv_name in input_convs:
            weight_name = input_conv_name + ".weight"
            try:
                if adapt_input_mode == "timm":
                    state_dict[weight_name] = adapt_input_conv(in_chans, state_dict[weight_name])
                    _logger.warning(
                        f"Converted input conv {input_conv_name} pretrained weights from 3 to {in_chans} channel(s) using timm strategy"
                    )
                else:
                    state_dict[weight_name] = my_adapt_input_conv(
                        in_chans, state_dict[weight_name], model_conv_weight=model.state_dict()[weight_name]
                    )
                    _logger.warning(
                        f"Converted input conv {input_conv_name} pretrained weights from 3 to {in_chans} channel(s) using custom strategy"
                    )
            except NotImplementedError as e:
                del state_dict[weight_name]
                strict = False
                _logger.warning(
                    f"Unable to convert pretrained {input_conv_name} weights, using random init for this layer."
                )

    classifiers = default_cfg.get("classifier", None)
    label_offset = default_cfg.get("label_offset", 0)
    if classifiers is not None:
        if isinstance(classifiers, str):
            classifiers = (classifiers,)
        if num_classes != default_cfg["num_classes"]:
            for classifier_name in classifiers:
                # completely discard fully connected if model num_classes doesn't match pretrained weights
                del state_dict[classifier_name + ".weight"]
                del state_dict[classifier_name + ".bias"]
            strict = False
        elif label_offset > 0:
            for classifier_name in classifiers:
                # special case for pretrained weights with an extra background class in pretrained weights
                classifier_weight = state_dict[classifier_name + ".weight"]
                state_dict[classifier_name + ".weight"] = classifier_weight[label_offset:]
                classifier_bias = state_dict[classifier_name + ".bias"]
                state_dict[classifier_name + ".bias"] = classifier_bias[label_offset:]

    model.load_state_dict(state_dict, strict=strict)
