import time
import torch

from tqdm import tqdm
from utils.functions import move_to
#from utils.model_utils import set_decode_type


def rollout(model, dataset, opts):
    # Put in greedy evaluation mode!
    #set_decode_type(model, "greedy")
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


def train_epoch(model, optimizer, baseline, lr_scheduler, epoch, dataset, tb_logger, opts):
    print("Start train epoch {}, lr={} for run {}".format(epoch, optimizer.param_groups[0]['lr'], opts['run_name']))
    step = epoch * (opts['epoch_size'] // opts['batch_size'])
    is_cuda = torch.cuda.is_available()
    start_time = time.time()
    if not opts['no_tensorboard']:
        tb_logger.log_value('learnrate_pg0', optimizer.param_groups[0]['lr'], step)
    
    # Put model in train mode and setup dataloader
    model.train()
    training_dataloader = training_dataloader = torch.utils.data.DataLoader(dataset, batch_size=opts['batch_size'])
    for batch_id, batch in enumerate(tqdm(training_dataloader, disable=opts['no_progress_bar'])):
        train_batch(model, optimizer, baseline, epoch, batch_id, step, batch, tb_logger, opts)
        step += 1

    epoch_duration = time.time() - start_time
    print("Finished epoch {}, took {} s".format(epoch, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))
    if is_cuda:
        torch.cuda.empty_cache()

    # lr_scheduler should be called at end of epoch
    lr_scheduler.step()


def train_batch(model, optimizer, baseline, epoch, batch_id, step, batch, tb_logger, opts):
    x = move_to(batch, opts['device'])