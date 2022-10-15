# rotation matrices, format (N, 3, 3).
from typing import Any, Iterator, List, Tuple, Union
import torch
from torch import device
from detectron2.utils.env import TORCH_VERSION

if TORCH_VERSION < (1, 8):
    _maybe_jit_unused = torch.jit.unused
else:

    def _maybe_jit_unused(x):
        return x


class Rots:
    """This structure stores a list of rotation matrices as a Nx3x3
    torch.Tensor. It supports some common methods about rots, and also behaves
    like a Tensor (support indexing, `to(device)`, `.device`, and iteration
    over all rots)

    Attributes:
        tensor (torch.Tensor): float matrix of Nx3x3.
    """

    def __init__(self, tensor: torch.Tensor):
        """
        Args:
            tensor (Tensor[float]):
                * a Nx3x3 matrix.
        """
        device = tensor.device if isinstance(tensor, torch.Tensor) else torch.device("cpu")
        tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device)
        if tensor.numel() == 0:
            # Use reshape, so we don't end up creating a new tensor that does not depend on
            # the inputs (and consequently confuses jit)
            tensor = torch.reshape(0, 3, 3).to(dtype=torch.float32, device=device)
        assert tensor.ndim == 3 and (tensor.shape[1:] == (3, 3)), tensor.shape

        self.tensor = tensor

    def clone(self) -> "Rots":
        """Clone the Rots.

        Returns:
            Rots
        """
        return Rots(self.tensor.clone())

    @_maybe_jit_unused
    def to(self, device: torch.device = None, **kwargs) -> "Rots":
        return Rots(self.tensor.to(device=device, **kwargs))

    def __getitem__(self, item: Union[int, slice, torch.BoolTensor]) -> "Rots":
        """
        Returns:
            Rots: Create a new :class:`Rots` by indexing.

        The following usage are allowed:
        1. `new_rots = rots[3]`: return a `Rots` which contains only one pose.
        2. `new_rots = rots[2:10]`: return a slice of rots.
        3. `new_rots = rots[vector]`, where vector is a torch.BoolTensor
           with `length = len(rots)`. Nonzero elements in the vector will be selected.

        Note that the returned Rots might share storage with this Rots,
        subject to Pytorch's indexing semantics.
        """
        if isinstance(item, int):
            return Rots(self.tensor[item].view(1, 3, 3))
        b = self.tensor[item]
        assert b.ndim == 3, "Indexing on Rots with {} failed!".format(item)
        return Rots(b)

    def __len__(self) -> int:
        return self.tensor.shape[0]

    def __repr__(self) -> str:
        return "Rots(" + str(self.tensor) + ")"

    @classmethod
    def cat(cls, rots_list: List["Rots"]) -> "Rots":
        """Concatenates a list of Rots into a single Rots.

        Arguments:
            rots_list (list[Rots])

        Returns:
            Rots: the concatenated Rots
        """
        if torch.jit.is_scripting():
            # https://github.com/pytorch/pytorch/issues/18627
            # 1. staticmethod can be used in torchscript, But we can not use
            # `type(xxx).staticmethod` because torchscript only supports function
            # `type` with input type `torch.Tensor`.
            # 2. classmethod is not fully supported by torchscript. We explicitly assign
            # cls to ThisClassName as a workaround to get torchscript support.
            cls = Rots
        assert isinstance(rots_list, (list, tuple))
        if len(rots_list) == 0:
            return cls(torch.empty(0))
        assert all(isinstance(pose, Rots) for pose in rots_list)

        # use torch.cat (v.s. layers.cat) so the returned tensor never share storage with input
        cat_rots = cls(rots_list[0])(torch.cat([p.tensor for p in rots_list], dim=0))
        return cat_rots

    @property
    def device(self) -> device:
        return self.tensor.device

    # type "Iterator[torch.Tensor]", yield, and iter() not supported by torchscript
    # https://github.com/pytorch/pytorch/issues/18627
    @torch.jit.unused
    def __iter__(self) -> Iterator[torch.Tensor]:
        """Yield a rot as a Tensor of shape (3,3) at a time."""
        yield from self.tensor
