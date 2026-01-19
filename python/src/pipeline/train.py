"""
Training Utilities for NGLab.

Provides high-level functions for rolling out models, training epochs,
and handling gradient clipping.
"""

import math
import os
import time
from typing import cast

import torch
from tqdm import tqdm
from utils.functions import move_to
from utils.log_utils import log_timeseries_values
from utils.model_utils import get_inner_model


def rollout(model, dataset, opts):
    """
    Perform evaluation rollout on a dataset.

    Args:
        model (nn.Module): The model to evaluate.
        dataset (Dataset): The dataset to roll out on.
        opts (Dict): Evaluation options.

    Returns:
        torch.Tensor: Concatenated results from the rollout.
    """
    # Put in greedy evaluation mode!
    # set_decode_type(model, "greedy")
    model.eval()

    def eval_model_bat(bat):
        """
        Evaluate a single batch.
        """
        with torch.no_grad():
            cost, _ = model(move_to(bat, opts["device"]))
        return cost.data.cpu()

    return torch.cat(
        [
            eval_model_bat(bat)
            for bat in tqdm(
                torch.utils.data.DataLoader(
                    dataset, batch_size=opts["eval_batch_size"], pin_memory=True
                ),
                disable=opts["no_progress_bar"],
            )
        ],
        0,
    )


def clip_grad_norms(param_groups, max_norm=math.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param optimizer:
    :param max_norm:
    :param gradient_norms_log:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group["params"],
            (
                max_norm if max_norm > 0 else math.inf
            ),  # Inf so no clipping but still call to calc
            norm_type=2,
        )
        for group in param_groups
    ]
    grad_norms_clipped = (
        [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    )
    return grad_norms, grad_norms_clipped


def train_epoch(
    model, optimizer, baseline, lr_scheduler, epoch, dataset, tb_logger, opts
):
    """
    Train the model for one epoch.
    """
    print(
        "Start train epoch {}, lr={} for run {}".format(
            epoch, optimizer.param_groups[0]["lr"], opts["run_name"]
        )
    )
    is_cuda = torch.cuda.is_available()
    start_time = time.time()
    if not opts["no_tensorboard"]:
        tb_logger.log_value("learnrate_pg0", optimizer.param_groups[0]["lr"], epoch)

    # Put model in train mode and setup dataloader
    step = epoch * opts["batch_size"]
    model.train()
    training_dataloader = training_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=opts["batch_size"]
    )
    for batch_id, batch in enumerate(
        tqdm(training_dataloader, disable=opts["no_progress_bar"])
    ):
        train_batch(
            model, optimizer, baseline, epoch, batch_id, batch, step, tb_logger, opts
        )
        step += 1

    epoch_duration = time.time() - start_time
    print(
        "Finished epoch {}, took {} s".format(
            epoch, time.strftime("%H:%M:%S", time.gmtime(epoch_duration))
        )
    )
    if is_cuda:
        torch.cuda.empty_cache()

    if (
        opts["checkpoint_epochs"] != 0 and epoch % opts["checkpoint_epochs"] == 0
    ) or epoch == opts["n_epochs"] - 1:
        print("Saving model and state...")
        torch.save(
            {
                "model": get_inner_model(model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all(),
                "baseline": baseline.state_dict(),
            },
            os.path.join(opts["save_dir"], f"epoch-{epoch}.pt"),
        )

    # lr_scheduler should be called at end of epoch
    lr_scheduler.step()


def train_batch(
    model, optimizer, baseline, epoch, batch_id, batch, step, tb_logger, opts
):
    """
    Train the model on a single batch.
    """
    batch = move_to(batch, opts["device"])
    x = batch["Price"]
    y = cast(torch.Tensor, batch["Labels"])

    # Compute output and loss
    output = model(x)
    # loss = baseline.loss(output, y)
    # Use MAE (L1 Loss) as requested for Polymarket
    loss = torch.nn.functional.l1_loss(output, y)

    # Perform backward pass and optimization step
    optimizer.zero_grad()
    loss.backward()

    # Clip gradient norms and get (clipped) gradient norms for logging
    grad_norms = clip_grad_norms(optimizer.param_groups, opts["max_grad_norm"])
    optimizer.step()

    # Logging
    if step % int(opts["log_step"]) == 0:
        log_timeseries_values(
            loss, grad_norms, epoch, batch_id, step, output, tb_logger, opts
        )
