import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Union
from torch import Tensor
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t


class MeanConv2d(nn.Conv2d):
    """Conv2d with weight centralization.

    ref: Weight and Gradient Centralization in Deep Neural Networks. https://arxiv.org/pdf/2010.00866.pdf
    """

    def forward(self, x):
        w = self.weight  # [c_out, c_in, k, k]
        w = w - torch.mean(w, dim=[1, 2, 3], keepdim=True)
        return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)


class MeanConvTranspose2d(nn.ConvTranspose2d):
    """ConvTranspose2d with Weight Centralization.

    Paper: `Micro-Batch Training with Batch-Channel Normalization and Weight Standardization` -
        https://arxiv.org/abs/1903.10520v2
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        output_padding: _size_2_t = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: int = 1,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
        eps=1e-6,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )
        self.eps = eps

    def forward(self, input: Tensor, output_size: Optional[List[int]] = None) -> Tensor:
        if self.padding_mode != "zeros":
            raise ValueError("Only `zeros` padding mode is supported for ConvTranspose2d")

        assert isinstance(self.padding, tuple)
        # One cannot replace List by Tuple or Sequence in "_output_padding" because
        # TorchScript does not support `Sequence[T]` or `Tuple[T, ...]`.
        output_padding = self._output_padding(
            input, output_size, self.stride, self.padding, self.kernel_size, self.dilation
        )  # type: ignore[arg-type]

        w = self.weight
        w = w - torch.mean(w, dim=[1, 2, 3], keepdim=True)
        return F.conv_transpose2d(
            input, w, self.bias, self.stride, self.padding, output_padding, self.groups, self.dilation
        )
