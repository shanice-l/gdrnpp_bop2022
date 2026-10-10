import os

import torch


def inference_device() -> torch.device:
    requested = os.environ.get("GDRN_DEVICE", "").strip().lower()
    if requested and requested not in {"cuda", "xpu", "cpu"}:
        raise ValueError("GDRN_DEVICE must be cuda, xpu, or cpu")

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("GDRN_DEVICE=cuda requested, but CUDA is unavailable")
        return torch.device("cuda")
    if requested == "xpu":
        if not hasattr(torch, "xpu") or not torch.xpu.is_available():
            raise RuntimeError("GDRN_DEVICE=xpu requested, but XPU is unavailable")
        return torch.device("xpu")
    if requested == "cpu":
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


def synchronize_device(device) -> None:
    device_type = torch.device(device).type
    if device_type == "cuda":
        torch.cuda.synchronize()
    elif device_type == "xpu":
        torch.xpu.synchronize()