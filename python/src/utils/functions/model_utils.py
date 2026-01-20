"""
Utilities for model manipulation and setup.
"""

import os
from typing import Any, cast

import torch
from torch import nn

from .functions import load_model


def set_decode_type(model: nn.Module, decode_type: str) -> None:
    """
    Set the decoding type for the model.

    Args:
        model (nn.Module): The model instance.
        decode_type (str): The decoding strategy (e.g., 'greedy', 'sampling').
    """
    if isinstance(model, nn.DataParallel):
        model = model.module
    if hasattr(model, "set_decode_type"):
        # Use cast(Any, ...) to avoid Mypy's confusion about custom methods on nn.Module
        cast(Any, model).set_decode_type(decode_type)


def get_inner_model(model: nn.Module) -> nn.Module:
    """
    Unwrap DataParallel model if necessary.

    Args:
        model (nn.Module): The model instance.

    Returns:
        nn.Module: The inner model.
    """
    return model.module if isinstance(model, nn.DataParallel) else model


def setup_model(
    name: str, general_path: str, device: torch.device, lock: Any | None = None
) -> nn.Module:
    """
    Setup and load a model from disk.

    Args:
        name (str): Model name or identifier.
        general_path (str): Base directory path.
        device (torch.device): Device to load model on.
        lock (threading.Lock): Optional lock for thread-safe loading.

    Returns:
        nn.Module: The loaded and initialized model.
    """

    def _load_model(
        general_path: str, model_path: str, device: torch.device, lock: Any | None
    ) -> nn.Module:
        model_path = os.path.join(general_path, model_path)
        if lock is not None:
            with lock:
                model, _ = load_model(model_path)
        else:
            model, _ = load_model(model_path)

        model.to(device)
        model.eval()
        return model

    return _load_model(general_path, name, device, lock)
