import warnings

import torch


# ----------------------------------------------------------------------------
# Replace NaN/Inf with specified numerical values.
# (from https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/torch_utils/misc.py)

try:
    nan_to_num = torch.nan_to_num  # 1.8.0a0
except AttributeError:

    def nan_to_num(input, nan=0.0, posinf=None, neginf=None, *, out=None):  # pylint: disable=redefined-builtin
        assert isinstance(input, torch.Tensor)
        if posinf is None:
            posinf = torch.finfo(input.dtype).max
        if neginf is None:
            neginf = torch.finfo(input.dtype).min
        assert nan == 0
        return torch.clamp(input.unsqueeze(0).nansum(0), min=neginf, max=posinf, out=out)


def set_nan_to_0(a, name=None, verbose=False):
    if torch.isnan(a).any():
        if verbose and name is not None:
            print("nan in {}".format(name))
        a[a != a] = 0
    return a


# ----------------------------------------------------------------------------
# Symbolic assert.

try:
    symbolic_assert = torch._assert  # 1.8.0a0 # pylint: disable=protected-access
except AttributeError:
    symbolic_assert = torch.Assert  # 1.7.0

# ----------------------------------------------------------------------------


class suppress_tracer_warnings(warnings.catch_warnings):
    """Context manager to suppress known warnings in torch.jit.trace()."""

    def __enter__(self):
        super().__enter__()
        warnings.simplefilter("ignore", category=torch.jit.TracerWarning)
        return self


# ----------------------------------------------------------------------------


def assert_shape(tensor, ref_shape):
    """Assert that the shape of a tensor matches the given list of integers.

    None indicates that the size of a dimension is allowed to vary.
    Performs symbolic assertion when used in torch.jit.trace().
    """
    if tensor.ndim != len(ref_shape):
        raise AssertionError(f"Wrong number of dimensions: got {tensor.ndim}, expected {len(ref_shape)}")
    for idx, (size, ref_size) in enumerate(zip(tensor.shape, ref_shape)):
        if ref_size is None:
            pass
        elif isinstance(ref_size, torch.Tensor):
            with suppress_tracer_warnings():  # as_tensor results are registered as constants
                symbolic_assert(
                    torch.equal(torch.as_tensor(size), ref_size),
                    f"Wrong size for dimension {idx}",
                )
        elif isinstance(size, torch.Tensor):
            with suppress_tracer_warnings():  # as_tensor results are registered as constants
                symbolic_assert(
                    torch.equal(size, torch.as_tensor(ref_size)),
                    f"Wrong size for dimension {idx}: expected {ref_size}",
                )
        elif size != ref_size:
            raise AssertionError(f"Wrong size for dimension {idx}: got {size}, expected {ref_size}")
