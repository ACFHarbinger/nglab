import os
import torch.nn as nn

from .functions import load_model


def set_decode_type(model, decode_type):
    if isinstance(model, nn.DataParallel):
        model = model.module
    model.set_decode_type(decode_type)


def get_inner_model(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def setup_model(name, general_path, device, lock=None):
    def _load_model(general_path, model_path, device, lock):
        model_path = os.path.join(general_path, model_path)
        with lock:
            model, _ = load_model(model_path)
        
        model.to(device)
        model.eval()
        return model

    return _load_model(general_path, name, device, lock)