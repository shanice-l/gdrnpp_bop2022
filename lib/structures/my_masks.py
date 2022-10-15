import copy
from typing import Any, Iterator, List, Union

import numpy as np
import pycocotools.mask as mask_utils
import torch

from detectron2.layers.roi_align import ROIAlign
from torchvision.ops import RoIPool
from detectron2.structures.masks import (
    PolygonMasks,
    polygons_to_bitmask,
)  # BitMasks


class MyBitMasks(object):
    """# NOTE: modified to support cpu_only option This class stores the
    segmentation masks for all objects in one image, in the form of bitmaps.

    Attributes:
        tensor: bool Tensor of N,H,W, representing N instances in the image.
    """

    def __init__(self, tensor: Union[torch.Tensor, np.ndarray], cpu_only: bool = True):
        """
        Args:
            tensor: bool Tensor of N,H,W, representing N instances in the image.
            cpu_only: keep the masks on cpu even when to(device) is called
        """
        device = tensor.device if isinstance(tensor, torch.Tensor) else torch.device("cpu")
        tensor = torch.as_tensor(tensor, dtype=torch.bool, device=device)
        assert tensor.dim() == 3, tensor.size()
        self.image_size = tensor.shape[1:]
        self.tensor = tensor
        self.cpu_only = cpu_only

    def to(self, device: str, **kwargs) -> "MyBitMasks":
        if not self.cpu_only:
            return MyBitMasks(self.tensor.to(device, **kwargs), cpu_only=False)
        else:
            return MyBitMasks(self.tensor.to("cpu", **kwargs), cpu_only=True)

    def to_device(self, device: str = "cuda", **kwargs) -> "MyBitMasks":
        # force to device
        return MyBitMasks(self.tensor.to(device, **kwargs), cpu_only=False)

    def crop_and_resize(
        self,
        boxes: torch.Tensor,
        mask_size: int,
        interpolation: str = "bilinear",
    ) -> torch.Tensor:
        """# NOTE: if self.cpu_only, convert boxes to cpu Crop each bitmask by
        the given box, and resize results to (mask_size, mask_size). This can
        be used to prepare training targets for Mask R-CNN. It has less
        reconstruction error compared to rasterization with polygons. However
        we observe no difference in accuracy, but MyBitMasks requires more
        memory to store all the masks.

        Args:
            boxes (Tensor): Nx4 tensor storing the boxes for each mask
            mask_size (int): the size of the rasterized mask.
            interpolation (str): bilinear | nearest

        Returns:
            Tensor:
                A bool tensor of shape (N, mask_size, mask_size), where
                N is the number of predicted boxes for this image.
        """
        assert len(boxes) == len(self), "{} != {}".format(len(boxes), len(self))
        if self.cpu_only:
            device = "cpu"
        else:
            device = self.tensor.device

        batch_inds = torch.arange(len(boxes), device=device).to(dtype=boxes.dtype)[:, None]
        rois = torch.cat([batch_inds, boxes.to(device)], dim=1)  # Nx5

        bit_masks = self.tensor.to(dtype=torch.float32)
        rois = rois.to(device=device)
        # on cpu, speed compared to cv2?
        if interpolation == "nearest":
            op = RoIPool((mask_size, mask_size), 1.0)
        elif interpolation == "bilinear":
            op = ROIAlign((mask_size, mask_size), 1.0, 0, aligned=True)
        else:
            raise ValueError(f"Unknown interpolation type: {interpolation}")
        output = op.forward(bit_masks[:, None, :, :], rois).squeeze(1)
        output = output >= 0.5
        return output

    def __getitem__(self, item: Union[int, slice, torch.BoolTensor]) -> "MyBitMasks":
        """
        Returns:
            MyBitMasks: Create a new :class:`MyBitMasks` by indexing.

        The following usage are allowed:

        1. `new_masks = masks[3]`: return a `MyBitMasks` which contains only one mask.
        2. `new_masks = masks[2:10]`: return a slice of masks.
        3. `new_masks = masks[vector]`, where vector is a torch.BoolTensor
           with `length = len(masks)`. Nonzero elements in the vector will be selected.

        Note that the returned object might share storage with this object,
        subject to Pytorch's indexing semantics.
        """
        if isinstance(item, int):
            return MyBitMasks(self.tensor[item].view(1, -1))
        m = self.tensor[item]
        assert m.dim() == 3, "Indexing on MyBitMasks with {} returns a tensor with shape {}!".format(item, m.shape)
        return MyBitMasks(m)

    def __iter__(self) -> torch.Tensor:
        yield from self.tensor

    def __repr__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "num_instances={})".format(len(self.tensor))
        return s

    def __len__(self) -> int:
        return self.tensor.shape[0]

    def nonempty(self) -> torch.Tensor:
        """Find masks that are non-empty.

        Returns:
            Tensor: a BoolTensor which represents
                whether each mask is empty (False) or non-empty (True).
        """
        return self.tensor.flatten(1).any(dim=1)

    @staticmethod
    def from_polygon_masks(
        polygon_masks: Union["PolygonMasks", List[List[np.ndarray]]],
        height: int,
        width: int,
        cpu_only: bool = True,
    ) -> "MyBitMasks":
        """
        Args:
            polygon_masks (list[list[ndarray]] or PolygonMasks)
            height, width (int)
        """
        if isinstance(polygon_masks, PolygonMasks):
            polygon_masks = polygon_masks.polygons
        masks = [polygons_to_bitmask(p, height, width) for p in polygon_masks]
        return MyBitMasks(
            torch.stack([torch.from_numpy(x) for x in masks]),
            cpu_only=cpu_only,
        )

    def get_bounding_boxes(self) -> None:
        # not needed now
        raise NotImplementedError
