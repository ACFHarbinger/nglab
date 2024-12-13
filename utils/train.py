import torch


def get_inner_model(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def train_epoch(model, optimizer, epoch, dataset, opts):
    return 0