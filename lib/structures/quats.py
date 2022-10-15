# quaternions
from typing import Any, Iterator, List, Tuple, Union

import torch
from torch import device
from detectron2.utils.env import TORCH_VERSION

if TORCH_VERSION < (1, 8):
    _maybe_jit_unused = torch.jit.unused
else:

    def _maybe_jit_unused(x):
        return x


class Quats:
    """This structure stores a list of quaternions as a Nx4 torch.Tensor. It
    supports some common methods about quats, and also behaves like a Tensor
    (support indexing, `to(device)`, `.device`, and iteration over all quats)

    Attributes:
        tensor: float matrix of Nx4.
    """

    def __init__(self, tensor: torch.Tensor):
        """
        Args:
            tensor (Tensor[float]):
                * a Nx4 matrix.  Each row is (qw, qx, qy, qz).
        """
        device = tensor.device if isinstance(tensor, torch.Tensor) else torch.device("cpu")
        tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device)
        if tensor.numel() == 0:
            # Use reshape, so we don't end up creating a new tensor that does not depend on
            # the inputs (and consequently confuses jit)
            tensor = torch.reshape(0, 4).to(dtype=torch.float32, device=device)
        assert tensor.ndim == 2 and (tensor.shape[-1] == 4), tensor.shape

        self.tensor = tensor

    def clone(self) -> "Quats":
        """Clone the Quats.

        Returns:
            Quats
        """
        return Quats(self.tensor.clone())

    @_maybe_jit_unused
    def to(self, device: torch.device = None, **kwargs) -> "Quats":
        return Quats(self.tensor.to(device=device, **kwargs))

    def __getitem__(self, item: Union[int, slice, torch.BoolTensor]) -> "Quats":
        """
        Returns:
            Quats: Create a new :class:`Quats` by indexing.

        The following usage are allowed:
        1. `new_quats = quats[3]`: return a `Quats` which contains only one quat.
        2. `new_quats = quats[2:10]`: return a slice of quats.
        3. `new_quats = quats[vector]`, where vector is a torch.BoolTensor
           with `length = len(quats)`. Nonzero elements in the vector will be selected.

        Note that the returned Quats might share storage with this Quats,
        subject to Pytorch's indexing semantics.
        """
        if isinstance(item, int):
            return Quats(self.tensor[item].view(1, -1))
        b = self.tensor[item]
        assert b.ndim == 2, "Indexing on Quats with {} failed to return a matrix!".format(item)
        return Quats(b)

    def __len__(self) -> int:
        return self.tensor.shape[0]

    def __repr__(self) -> str:
        return "Quats(" + str(self.tensor) + ")"

    @classmethod
    def cat(cls, quats_list: List["Quats"]) -> "Quats":
        """Concatenates a list of Quats into a single Quats.

        Arguments:
            quats_list (list[Quats])

        Returns:
            Quats: the concatenated Quats
        """
        if torch.jit.is_scripting():
            # https://github.com/pytorch/pytorch/issues/18627
            # 1. staticmethod can be used in torchscript, But we can not use
            # `type(quats).staticmethod` because torchscript only supports function
            # `type` with input type `torch.Tensor`.
            # 2. classmethod is not fully supported by torchscript. We explicitly assign
            # cls to Quats as a workaround to get torchscript support.
            cls = Quats
        assert isinstance(quats_list, (list, tuple))
        if len(quats_list) == 0:
            return cls(torch.empty(0))
        assert all(isinstance(quats, Quats) for quats in quats_list)

        # use torch.cat (v.s. layers.cat) so the returned quats never share storage with input
        cat_quats = cls(quats_list[0])(torch.cat([q.tensor for q in quats_list], dim=0))
        return cat_quats

    @property
    def device(self) -> device:
        return self.tensor.device

    # type "Iterator[torch.Tensor]", yield, and iter() not supported by torchscript
    # https://github.com/pytorch/pytorch/issues/18627
    @torch.jit.unused
    def __iter__(self) -> Iterator[torch.Tensor]:
        """Yield a quat as a Tensor of shape (4,) at a time."""
        yield from self.tensor
