import torch

from tqdm import tqdm
from utils.functions import move_to
from utils.model_utils import set_decode_type


def rollout(model, dataset, opts):
    # Put in greedy evaluation mode!
    set_decode_type(model, "greedy")
    model.eval()

    def eval_model_bat(bat):
        with torch.no_grad():
            cost, _ = model(move_to(bat, opts['device']))
        return cost.data.cpu()

    return torch.cat([
        eval_model_bat(bat)
        for bat
        in tqdm(torch.utils.data.DataLoader(dataset, batch_size=opts['eval_batch_size'], pin_memory=True), disable=opts['no_progress_bar'])
    ], 0)


def train_epoch(model, optimizer, epoch, dataset, opts):
    return 0