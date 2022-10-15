# translations
from typing import Any, Iterator, List, Tuple, Union

import torch
from torch import device
from detectron2.utils.env import TORCH_VERSION

if TORCH_VERSION < (1, 8):
    _maybe_jit_unused = torch.jit.unused
else:

    def _maybe_jit_unused(x):
        return x


class Translations:
    """This structure stores a list of translations a Nx3 torch.Tensor.

    Attributes:
        tensor: float matrix of Nx3.
    """

    def __init__(self, tensor: torch.Tensor):
        """
        Args:
            tensor (Tensor[float]):
                * a Nx3 matrix.  Each row is (tx, ty, tz).
        """
        device = tensor.device if isinstance(tensor, torch.Tensor) else torch.device("cpu")
        tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device)
        if tensor.numel() == 0:
            # Use reshape, so we don't end up creating a new tensor that does not depend on
            # the inputs (and consequently confuses jit)
            tensor = torch.reshape(0, 3).to(dtype=torch.float32, device=device)
        assert tensor.ndim == 2 and (tensor.shape[-1] == 3), tensor.shape

        self.tensor = tensor

    def clone(self) -> "Translations":
        """Clone the Translations.

        Returns:
            Translations
        """
        return Translations(self.tensor.clone())

    @_maybe_jit_unused
    def to(self, device: torch.device = None, **kwargs) -> "Translations":
        return Translations(self.tensor.to(device=device, **kwargs))

    def __getitem__(self, item: Union[int, slice, torch.BoolTensor]) -> "Translations":
        """
        Returns:
            Translations: Create a new :class:`Translations` by indexing.

        The following usage are allowed:
        1. `new_transes = transes[3]`: return a `Translations` which contains only one translation.
        2. `new_transes = transes[2:10]`: return a slice of transes.
        3. `new_transes = transes[vector]`, where vector is a torch.BoolTensor
           with `length = len(transes)`. Nonzero elements in the vector will be selected.

        Note that the returned Translations might share storage with this Translations,
        subject to Pytorch's indexing semantics.
        """
        if isinstance(item, int):
            return Translations(self.tensor[item].view(1, -1))
        b = self.tensor[item]
        assert b.ndim == 2, "Indexing on Translations with {} failed to return a matrix!".format(item)
        return Translations(b)

    def __len__(self) -> int:
        return self.tensor.shape[0]

    def __repr__(self) -> str:
        return "Translations(" + str(self.tensor) + ")"

    def get_centers_2d(self, K: torch.Tensor) -> torch.Tensor:
        """
        Args:
            K: camera intrinsic matrices, Nx3x3 or 1x3x3
        Returns:
            The 2d projected object centers in a Nx2 array of (x, y).
        """
        bs = self.tensor.shape[0]
        proj = (K @ self.tensor.view(bs, 3, 1)).view(bs, 3)
        centers_2d = proj[:, :2] / proj[:, 2:3]  # Nx2
        return centers_2d

    @classmethod
    def cat(transes_list: List["Translations"]) -> "Translations":
        """Concatenates a list of Translations into a single Translations.

        Arguments:
            transes_list (list[Translations])

        Returns:
            Translations: the concatenated Translations
        """
        if torch.jit.is_scripting():
            # https://github.com/pytorch/pytorch/issues/18627
            # 1. staticmethod can be used in torchscript, But we can not use
            # `type(transes).staticmethod` because torchscript only supports function
            # `type` with input type `torch.Tensor`.
            # 2. classmethod is not fully supported by torchscript. We explicitly assign
            # cls to Translations as a workaround to get torchscript support.
            cls = Translations
        assert isinstance(transes_list, (list, tuple))
        if len(transes_list) == 0:
            return cls(torch.empty(0))
        assert all(isinstance(transes, Translations) for transes in transes_list)

        # use torch.cat (v.s. layers.cat) so the returned transes never share storage with input
        cat_transes = cls(transes_list[0])(torch.cat([t.tensor for t in transes_list], dim=0))
        return cat_transes

    @property
    def device(self) -> device:
        return self.tensor.device

    # type "Iterator[torch.Tensor]", yield, and iter() not supported by torchscript
    # https://github.com/pytorch/pytorch/issues/18627
    @torch.jit.unused
    def __iter__(self) -> Iterator[torch.Tensor]:
        """Yield a translation as a Tensor of shape (3,) at a time."""
        yield from self.tensor
