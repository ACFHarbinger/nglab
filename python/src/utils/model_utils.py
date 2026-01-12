"""
Utilities for model manipulation and setup.
"""
import os
import torch.nn as nn

from .functions import load_model


def set_decode_type(model, decode_type):
    """
    Set the decoding type for the model.

    Args:
        model (nn.Module): The model instance.
        decode_type (str): The decoding strategy (e.g., 'greedy', 'sampling').
    """
    if isinstance(model, nn.DataParallel):
        model = model.module
    model.set_decode_type(decode_type)


def get_inner_model(model):
    """
    Unwrap DataParallel model if necessary.

    Args:
        model (nn.Module): The model instance.

    Returns:
        nn.Module: The inner model.
    """
    return model.module if isinstance(model, nn.DataParallel) else model


def setup_model(name, general_path, device, lock=None):
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
    def _load_model(general_path, model_path, device, lock):
        model_path = os.path.join(general_path, model_path)
        with lock:
            model, _ = load_model(model_path)
        
        model.to(device)
        model.eval()
        return model

    return _load_model(general_path, name, device, lock)