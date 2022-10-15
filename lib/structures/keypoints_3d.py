# for example: store bbox3d+center3d => (N,9,3)
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


class Keypoints3Ds:
    """Modified from class Keypoints.

    Stores 3d keypoint annotation data. GT Instances have a
    `gt_3d_keypoints` property containing the x,y,z location of each
    keypoint. This tensor has shape (N, K, 3) where N is the number of
    instances and K is the number of keypoints per instance.
    """

    def __init__(self, keypoints: Union[torch.Tensor, np.ndarray, List[List[float]]]):
        """
        Arguments:
            keypoints: A Tensor, numpy array, or list of the x, y, z of each keypoint.
                The shape should be (N, K, 3) where N is the number of
                instances, and K is the number of keypoints per instance.
        """
        device = keypoints.device if isinstance(keypoints, torch.Tensor) else torch.device("cpu")
        keypoints = torch.as_tensor(keypoints, dtype=torch.float32, device=device)
        assert keypoints.ndim == 3 and keypoints.shape[2] == 3, keypoints.shape
        self.tensor = keypoints

    def __len__(self) -> int:
        return self.tensor.shape[0]

    def clone(self) -> "Keypoints3Ds":
        """Clone the Keypoints3Ds.

        Returns:
            Keypoints3Ds
        """
        return Keypoints3Ds(self.tensor.clone())

    def to(self, *args: Any, **kwargs: Any) -> "Keypoints3Ds":
        return type(self)(self.tensor.to(*args, **kwargs))

    def __getitem__(self, item: Union[int, slice, torch.BoolTensor]) -> "Keypoints3Ds":
        """Create a new `Keypoints3Ds` by indexing on this `Keypoints3Ds`.

        The following usage are allowed:

        1. `new_kpts = kpts[3]`: return a `Keypoints3Ds` which contains only one instance.
        2. `new_kpts = kpts[2:10]`: return a slice of key points.
        3. `new_kpts = kpts[vector]`, where vector is a torch.ByteTensor
           with `length = len(kpts)`. Nonzero elements in the vector will be selected.

        Note that the returned Keypoints3Ds might share storage with this Keypoints3Ds,
        subject to Pytorch's indexing semantics.
        """
        if isinstance(item, int):
            return Keypoints3Ds([self.tensor[item]])
        return Keypoints3Ds(self.tensor[item])

    def __repr__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "num_instances={})".format(len(self.tensor))
        return s

    @classmethod
    def cat(cls, keypoints3ds_list: List["Keypoints3Ds"]) -> "Keypoints3Ds":
        """Concatenates a list of Keypoints3Ds into a single Keypoints3Ds.

        Arguments:
            keypoints3ds_list (list[Keypoints3Ds])

        Returns:
            Keypoints3Ds: the concatenated Keypoints3Ds
        """
        if torch.jit.is_scripting():
            # https://github.com/pytorch/pytorch/issues/18627
            # 1. staticmethod can be used in torchscript, But we can not use
            # `type(keypoits3ds).staticmethod` because torchscript only supports function
            # `type` with input type `torch.Tensor`.
            # 2. classmethod is not fully supported by torchscript. We explicitly assign
            # cls to Keypoints3Ds as a workaround to get torchscript support.
            cls = Keypoints3Ds
        assert isinstance(keypoints3ds_list, (list, tuple))
        if len(keypoints3ds_list) == 0:
            return cls(torch.empty(0))
        assert all(isinstance(keypoints3ds, Keypoints3Ds) for keypoints3ds in keypoints3ds_list)

        # use torch.cat (v.s. layers.cat) so the returned tensor never share storage with input
        cat_keypoints3ds = type(keypoints3ds_list[0])(torch.cat([b.tensor for b in keypoints3ds_list], dim=0))
        return cat_keypoints3ds

    @property
    def device(self) -> device:
        return self.tensor.device

    # type "Iterator[torch.Tensor]", yield, and iter() not supported by torchscript
    # https://github.com/pytorch/pytorch/issues/18627
    @torch.jit.unused
    def __iter__(self) -> Iterator[torch.Tensor]:
        """Yield a 2d center as a Tensor of shape (2,) at a time."""
        yield from self.tensor
