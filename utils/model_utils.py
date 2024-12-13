import torch.nn as nn


def set_decode_type(model, decode_type):
    if isinstance(model, nn.DataParallel):
        model = model.module
    model.set_decode_type(decode_type)


def get_inner_model(model):
    return model.module if isinstance(model, nn.DataParallel) else model