# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch


def get_model_device(model: torch.nn.Module) -> torch.device:
    """Return the device of the first model parameter."""
    return next(model.parameters()).device


def empty_cache(device: torch.device) -> None:
    """Clear GPU cache for CUDA/ROCm backends."""
    if device.type == "cuda":
        torch.cuda.empty_cache()
