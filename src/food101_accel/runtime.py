"""CUDA precision, compilation, seeding, and environment helpers."""

from __future__ import annotations

import contextlib
import os
import platform
import random
from typing import Any, ContextManager

import torch
from torch import nn

from .config import ExperimentConfig


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return torch.device(requested)


def configure_precision(config: ExperimentConfig, device: torch.device) -> None:
    """Set an explicit numerical policy so baseline and accelerated runs differ cleanly."""

    torch.set_float32_matmul_precision(config.matmul_precision)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = config.allow_tf32
        torch.backends.cudnn.allow_tf32 = config.allow_tf32


def amp_dtype(config: ExperimentConfig) -> torch.dtype:
    return torch.bfloat16 if config.amp_dtype == "bfloat16" else torch.float16


def validate_acceleration_support(config: ExperimentConfig, device: torch.device) -> None:
    if config.profile != "accelerated":
        return
    if device.type != "cuda":
        raise RuntimeError("the accelerated profile requires a CUDA device")
    if config.amp_dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "this GPU does not support BF16; rerun with --amp-dtype float16"
        )


def autocast_context(
    config: ExperimentConfig, device: torch.device
) -> ContextManager[Any]:
    if not config.use_amp:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=amp_dtype(config))


def build_grad_scaler(config: ExperimentConfig) -> torch.amp.GradScaler:
    # BF16 has FP32's exponent range and normally does not need loss scaling.
    enabled = config.use_amp and config.amp_dtype == "float16"
    return torch.amp.GradScaler("cuda", enabled=enabled)


def maybe_compile(model: nn.Module, config: ExperimentConfig) -> nn.Module:
    if not config.use_compile:
        return model
    # Compiler errors remain visible because eager fallback would invalidate
    # the compiled profile label.
    return torch.compile(
        model,
        mode=config.compile_mode,
        fullgraph=False,
        dynamic=False,
    )


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def environment_metadata(device: torch.device) -> dict[str, Any]:
    cuda_metadata: dict[str, Any] | None = None
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        cuda_metadata = {
            "device_name": torch.cuda.get_device_name(device),
            "device_capability": list(torch.cuda.get_device_capability(device)),
            "total_memory_bytes": properties.total_memory,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "bf16_supported": torch.cuda.is_bf16_supported(),
        }
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": _torchvision_version(),
        "device": str(device),
        "cuda": cuda_metadata,
        "cuda_launch_blocking": os.environ.get("CUDA_LAUNCH_BLOCKING"),
    }


def _torchvision_version() -> str:
    import torchvision

    return torchvision.__version__
