import copy
from typing import Any, Iterator, List, Union

import numpy as np
import torch

from detectron2.layers.roi_align import ROIAlign
from torchvision.ops import RoIPool


class MyMaps(object):
    """# NOTE: This class stores the maps (NOCS, coordinates map, pvnet vector
    maps, offset maps, heatmaps) for all objects in one image, support cpu_only
    option.

    Attributes:
        tensor: bool Tensor of N,C,H,W, representing N instances in the image.
    """

    def __init__(self, tensor: Union[torch.Tensor, np.ndarray], cpu_only: bool = True):
        """
        Args:
            tensor: float Tensor of N,C,H,W, representing N instances in the image.
            cpu_only: keep the maps on cpu even when to(device) is called
        """
        device = tensor.device if isinstance(tensor, torch.Tensor) else torch.device("cpu")
        tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device)
        assert tensor.dim() == 4, tensor.size()
        self.image_size = tensor.shape[-2:]
        self.tensor = tensor
        self.cpu_only = cpu_only

    def to(self, device: str, **kwargs) -> "MyMaps":
        if not self.cpu_only:
            return MyMaps(self.tensor.to(device, **kwargs), cpu_only=False)
        else:
            return MyMaps(self.tensor.to("cpu", **kwargs), cpu_only=True)

    def to_device(self, device: str = "cuda", **kwargs) -> "MyMaps":
        # force to device
        return MyMaps(self.tensor.to(device, **kwargs), cpu_only=False)

    def crop_and_resize(
        self,
        boxes: torch.Tensor,
        map_size: int,
        interpolation: str = "bilinear",
    ) -> torch.Tensor:
        """# NOTE: if self.cpu_only, convert boxes to cpu
        Crop each map by the given box, and resize results to (map_size, map_size).
        This can be used to prepare training targets.
        Args:
            boxes (Tensor): Nx4 tensor storing the boxes for each map
            map_size (int): the size of the rasterized map.
            interpolation (str): bilinear | nearest

        Returns:
            Tensor:
                A bool tensor of shape (N, C, map_size, map_size), where
                N is the number of predicted boxes for this image.
        """
        assert len(boxes) == len(self), "{} != {}".format(len(boxes), len(self))
        if self.cpu_only:
            device = "cpu"
        else:
            device = self.tensor.device

        batch_inds = torch.arange(len(boxes), device=device).to(dtype=boxes.dtype)[:, None]
        rois = torch.cat([batch_inds, boxes.to(device)], dim=1)  # Nx5

        maps = self.tensor.to(dtype=torch.float32)
        rois = rois.to(device=device)
        # on cpu, speed compared to cv2?
        if interpolation == "nearest":
            op = RoIPool((map_size, map_size), 1.0)
        elif interpolation == "bilinear":
            op = ROIAlign((map_size, map_size), 1.0, 0, aligned=True)
        else:
            raise ValueError(f"Unknown interpolation type: {interpolation}")
        output = op.forward(maps, rois)
        return output

    def __getitem__(self, item: Union[int, slice, torch.BoolTensor]) -> "MyMaps":
        """
        Returns:
            MyMaps: Create a new :class:`MyMaps` by indexing.

        The following usage are allowed:

        1. `new_maps = maps[3]`: return a `MyMaps` which contains only one map.
        2. `new_maps = maps[2:10]`: return a slice of maps.
        3. `new_maps = maps[vector]`, where vector is a torch.BoolTensor
           with `length = len(maps)`. Nonzero elements in the vector will be selected.

        Note that the returned object might share storage with this object,
        subject to Pytorch's indexing semantics.
        """
        if isinstance(item, int):
            return MyMaps(self.tensor[item].view(1, -1))
        m = self.tensor[item]
        assert m.dim() == 4, "Indexing on MyMaps with {} returns a tensor with shape {}!".format(item, m.shape)
        return MyMaps(m)

    def __iter__(self) -> torch.Tensor:
        yield from self.tensor

    def __repr__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "num_instances={})".format(len(self.tensor))
        return s

    def __len__(self) -> int:
        return self.tensor.shape[0]

    def nonempty(self) -> torch.Tensor:
        """Find maps that are non-empty.

        Returns:
            Tensor: a BoolTensor which represents
                whether each map is empty (False) or non-empty (True).
        """
        return self.tensor.flatten(1).any(dim=1)
