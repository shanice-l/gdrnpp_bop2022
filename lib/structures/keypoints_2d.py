# for example: store projected bbox3d+center3d => (N,9,2)
from typing import Any, Iterator, List, Tuple, Union

import numpy as np
import torch
from torch import device
from detectron2.utils.env import TORCH_VERSION

if TORCH_VERSION < (1, 8):
    _maybe_jit_unused = torch.jit.unused
else:

    def _maybe_jit_unused(x):
        return x


class Keypoints2Ds:
    """Modified from class Keypoints.

    Stores 2d keypoint annotation data. GT Instances have a
    `gt_2d_keypoints` property containing the x,y location of each
    keypoint. This tensor has shape (N, K, 2) where N is the number of
    instances and K is the number of keypoints per instance.
    """

    def __init__(self, keypoints: Union[torch.Tensor, np.ndarray, List[List[float]]]):
        """
        Arguments:
            keypoints: A Tensor, numpy array, or list of the x, y of each keypoint.
                The shape should be (N, K, 2) where N is the number of
                instances, and K is the number of keypoints per instance.
        """
        device = keypoints.device if isinstance(keypoints, torch.Tensor) else torch.device("cpu")
        keypoints = torch.as_tensor(keypoints, dtype=torch.float32, device=device)
        assert keypoints.ndim == 3 and keypoints.shape[2] == 2, keypoints.shape
        self.tensor = keypoints

    def __len__(self) -> int:
        return self.tensor.shape[0]

    def clone(self) -> "Keypoints2Ds":
        """Clone the Keypoints2Ds.

        Returns:
            Keypoints2Ds
        """
        return Keypoints2Ds(self.tensor.clone())

    def to(self, *args: Any, **kwargs: Any) -> "Keypoints2Ds":
        return type(self)(self.tensor.to(*args, **kwargs))

    def to_heatmap(self, boxes: torch.Tensor, heatmap_size: int) -> torch.Tensor:
        # TODO: convert 2d keypoints to heatmap as proposed in Integoral Regression
        # copy from d2 if needed
        raise NotImplementedError

    def __getitem__(self, item: Union[int, slice, torch.BoolTensor]) -> "Keypoints2Ds":
        """Create a new `Keypoints2Ds` by indexing on this `Keypoints2Ds`.

        The following usage are allowed:

        1. `new_kpts = kpts[3]`: return a `Keypoints2Ds` which contains only one instance.
        2. `new_kpts = kpts[2:10]`: return a slice of key points.
        3. `new_kpts = kpts[vector]`, where vector is a torch.ByteTensor
           with `length = len(kpts)`. Nonzero elements in the vector will be selected.

        Note that the returned Keypoints2Ds might share storage with this Keypoints2Ds,
        subject to Pytorch's indexing semantics.
        """
        if isinstance(item, int):
            return Keypoints2Ds([self.tensor[item]])
        return Keypoints2Ds(self.tensor[item])

    def __repr__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "num_instances={})".format(len(self.tensor))
        return s

    @classmethod
    def cat(cls, keypoints2ds_list: List["Keypoints2Ds"]) -> "Keypoints2Ds":
        """Concatenates a list of Keypoints2Ds into a single Keypoints2Ds.

        Arguments:
            keypoints2ds_list (list[Keypoints2Ds])

        Returns:
            Keypoints2Ds: the concatenated Keypoints2Ds
        """
        if torch.jit.is_scripting():
            # https://github.com/pytorch/pytorch/issues/18627
            # 1. staticmethod can be used in torchscript, But we can not use
            # `type(xxx).staticmethod` because torchscript only supports function
            # `type` with input type `torch.Tensor`.
            # 2. classmethod is not fully supported by torchscript. We explicitly assign
            # cls to ThisClassName as a workaround to get torchscript support.
            cls = Keypoints2Ds
        assert isinstance(keypoints2ds_list, (list, tuple))
        if len(keypoints2ds_list) == 0:
            return cls(torch.empty(0))
        assert all(isinstance(keypoints2ds, Keypoints2Ds) for keypoints2ds in keypoints2ds_list)

        # use torch.cat (v.s. layers.cat) so the returned tensor never share storage with input
        cat_keypoints2ds = type(keypoints2ds_list[0])(torch.cat([b.tensor for b in keypoints2ds_list], dim=0))
        return cat_keypoints2ds

    @property
    def device(self) -> device:
        return self.tensor.device

    # type "Iterator[torch.Tensor]", yield, and iter() not supported by torchscript
    # https://github.com/pytorch/pytorch/issues/18627
    @torch.jit.unused
    def __iter__(self) -> Iterator[torch.Tensor]:
        """Yield a 2d center as a Tensor of shape (2,) at a time."""
        yield from self.tensor
