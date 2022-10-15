from typing import Any, Iterator, List, Tuple, Union

import torch
from torch import device
from detectron2.utils.env import TORCH_VERSION

if TORCH_VERSION < (1, 8):
    _maybe_jit_unused = torch.jit.unused
else:

    def _maybe_jit_unused(x):
        return x


class Center2Ds:
    """This structure stores a list of 2d centers (object/bbox centers) a Nx2
    torch.Tensor.

    Attributes:
        tensor: float matrix of Nx2.
    """

    def __init__(self, tensor: torch.Tensor):
        """
        Args:
            tensor (Tensor[float]):
                * a Nx2 matrix.  Each row is (x, y).
        """
        device = tensor.device if isinstance(tensor, torch.Tensor) else torch.device("cpu")
        tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device)
        if tensor.numel() == 0:
            # Use reshape, so we don't end up creating a new tensor that does not depend on
            # the inputs (and consequently confuses jit)
            tensor = torch.reshape(0, 2).to(dtype=torch.float32, device=device)
        assert tensor.ndim == 2 and (tensor.shape[-1] == 2), tensor.shape

        self.tensor = tensor

    def clone(self) -> "Center2Ds":
        """Clone the Center2Ds.

        Returns:
            Center2Ds
        """
        return Center2Ds(self.tensor.clone())

    @_maybe_jit_unused
    def to(self, device: torch.device = None, **kwargs) -> "Center2Ds":
        return Center2Ds(self.tensor.to(device=device, **kwargs))

    def __getitem__(self, item: Union[int, slice, torch.BoolTensor]) -> "Center2Ds":
        """
        Returns:
            Center2Ds: Create a new :class:`Center2Ds` by indexing.

        The following usage are allowed:
        1. `new_center2ds = center2ds[3]`: return a `Center2Ds` which contains only one 2d center.
        2. `new_center2ds = center2ds[2:10]`: return a slice of center2ds.
        3. `new_center2ds = center2ds[vector]`, where vector is a torch.BoolTensor
           with `length = len(center2ds)`. Nonzero elements in the vector will be selected.

        Note that the returned Center2Ds might share storage with this Center2Ds,
        subject to Pytorch's indexing semantics.
        """
        if isinstance(item, int):
            return Center2Ds(self.tensor[item].view(1, -1))
        b = self.tensor[item]
        assert b.ndim == 2, "Indexing on Center2Ds with {} failed to return a matrix!".format(item)
        return Center2Ds(b)

    def __len__(self) -> int:
        return self.tensor.shape[0]

    def __repr__(self) -> str:
        return "Center2Ds(" + str(self.tensor) + ")"

    @classmethod
    def cat(center2ds_list: List["Center2Ds"]) -> "Center2Ds":
        """Concatenates a list of Center2Ds into a single Center2Ds.

        Arguments:
            center2ds_list (list[Center2Ds])

        Returns:
            Center2Ds: the concatenated Center2Ds
        """
        if torch.jit.is_scripting():
            # https://github.com/pytorch/pytorch/issues/18627
            # 1. staticmethod can be used in torchscript, But we can not use
            # `type(center2ds).staticmethod` because torchscript only supports function
            # `type` with input type `torch.Tensor`.
            # 2. classmethod is not fully supported by torchscript. We explicitly assign
            # cls to Center2Ds as a workaround to get torchscript support.
            cls = Center2Ds
        assert isinstance(center2ds_list, (list, tuple))
        if len(center2ds_list) == 0:
            return cls(torch.empty(0))
        assert all(isinstance(center2ds, Center2Ds) for center2ds in center2ds_list)

        # use torch.cat (v.s. layers.cat) so the returned tensor never share storage with input
        cat_center2ds = cls(center2ds_list[0])(torch.cat([b.tensor for b in center2ds_list], dim=0))
        return cat_center2ds

    @property
    def device(self) -> device:
        return self.tensor.device

    # type "Iterator[torch.Tensor]", yield, and iter() not supported by torchscript
    # https://github.com/pytorch/pytorch/issues/18627
    @torch.jit.unused
    def __iter__(self) -> Iterator[torch.Tensor]:
        """Yield a 2d center as a Tensor of shape (2,) at a time."""
        yield from self.tensor
