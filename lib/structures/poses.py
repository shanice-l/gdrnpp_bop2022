# poses in [rot|trans] format (N, 3, 4).
from typing import Any, Iterator, List, Tuple, Union
import torch
from torch import device
from detectron2.utils.env import TORCH_VERSION

if TORCH_VERSION < (1, 8):
    _maybe_jit_unused = torch.jit.unused
else:

    def _maybe_jit_unused(x):
        return x


class Poses:
    """This structure stores a list of 6d poses as a Nx3x4 ([rot|trans]
    torch.Tensor. It supports some common methods about poses, and also behaves
    like a Tensor (support indexing, `to(device)`, `.device`, and iteration
    over all 6d poses)

    Attributes:
        tensor (torch.Tensor): float matrix of Nx3x4.
    """

    def __init__(self, tensor: torch.Tensor):
        """
        Args:
            tensor (Tensor[float]):
                * a Nx3x4 matrix.
        """
        device = tensor.device if isinstance(tensor, torch.Tensor) else torch.device("cpu")
        tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device)
        if tensor.numel() == 0:
            # Use reshape, so we don't end up creating a new tensor that does not depend on
            # the inputs (and consequently confuses jit)
            tensor = torch.reshape(0, 3, 4).to(dtype=torch.float32, device=device)
        assert tensor.ndim == 3 and (tensor.shape[1:] == (3, 4)), tensor.shape

        self.tensor = tensor

    def clone(self) -> "Poses":
        """Clone the Poses.

        Returns:
            Poses
        """
        return Poses(self.tensor.clone())

    @_maybe_jit_unused
    def to(self, device: torch.device = None, **kwargs) -> "Poses":
        return Poses(self.tensor.to(device=device, **kwargs))

    def __getitem__(self, item: Union[int, slice, torch.BoolTensor]) -> "Poses":
        """
        Returns:
            Poses: Create a new :class:`Poses` by indexing.

        The following usage are allowed:
        1. `new_poses = poses[3]`: return a `Poses` which contains only one pose.
        2. `new_poses = poses[2:10]`: return a slice of poses.
        3. `new_poses = poses[vector]`, where vector is a torch.BoolTensor
           with `length = len(poses)`. Nonzero elements in the vector will be selected.

        Note that the returned Poses might share storage with this Poses,
        subject to Pytorch's indexing semantics.
        """
        if isinstance(item, int):
            return Poses(self.tensor[item].view(1, 3, 4))
        b = self.tensor[item]
        assert b.ndim == 3, "Indexing on Poses with {} failed!".format(item)
        return Poses(b)

    def __len__(self) -> int:
        return self.tensor.shape[0]

    def __repr__(self) -> str:
        return "Poses(" + str(self.tensor) + ")"

    def get_centers_2d(self, K: torch.Tensor) -> torch.Tensor:
        """
        Args:
            K: camera intrinsic matrices, 1x3x3 or Nx3x3
        Returns:
            The 2d projected object centers in a Nx2 array of (x, y).
        """
        assert K.ndim == 3, K.shape
        bs = self.tensor.shape[0]
        proj = (K @ self.tensor[:, :3, [3]]).view(bs, 3)  # Nx3
        centers_2d = proj[:, :2] / proj[:, 2:3]  # Nx2
        return centers_2d

    @classmethod
    def cat(cls, poses_list: List["Poses"]) -> "Poses":
        """Concatenates a list of Poses into a single Poses.

        Arguments:
            poses_list (list[Poses])

        Returns:
            Poses: the concatenated Poses
        """
        if torch.jit.is_scripting():
            # https://github.com/pytorch/pytorch/issues/18627
            # 1. staticmethod can be used in torchscript, But we can not use
            # `type(poses).staticmethod` because torchscript only supports function
            # `type` with input type `torch.Tensor`.
            # 2. classmethod is not fully supported by torchscript. We explicitly assign
            # cls to Poses as a workaround to get torchscript support.
            cls = Poses
        assert isinstance(poses_list, (list, tuple))
        if len(poses_list) == 0:
            return cls(torch.empty(0))
        assert all(isinstance(pose, Poses) for pose in poses_list)

        # use torch.cat (v.s. layers.cat) so the returned poses never share storage with input
        cat_poses = cls(poses_list[0])(torch.cat([p.tensor for p in poses_list], dim=0))
        return cat_poses

    @property
    def device(self) -> device:
        return self.tensor.device

    # type "Iterator[torch.Tensor]", yield, and iter() not supported by torchscript
    # https://github.com/pytorch/pytorch/issues/18627
    @torch.jit.unused
    def __iter__(self) -> Iterator[torch.Tensor]:
        """Yield a 6d pose as a Tensor of shape (3,4) at a time."""
        yield from self.tensor
