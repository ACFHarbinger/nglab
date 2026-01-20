"""
Logging utilities for training, evaluation, and simulation.
"""

import datetime
import json
import os
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import wandb


def log_timeseries_values(  # noqa: PLR0913
    loss: float,
    grad_norms: tuple[list[float], list[float]],
    epoch: int,
    batch_id: int,
    step: int,
    output: torch.Tensor,
    tb_logger: Any,
    opts: dict[str, Any],
) -> None:
    """
    Log training metrics to console, TensorBoard, and WandB.
    """
    grad_norms_all, grad_norms_clipped = grad_norms

    # Log values to screen
    if step % (opts.get("log_step", 1) * 10) == 0:
        print(f"epoch: {epoch}, train_batch_id: {batch_id}, loss: {loss:.6f}")
        print(
            f"grad_norm: {grad_norms_all[0]:.6f}, clipped: {grad_norms_clipped[0]:.6f}"
        )

    # Log values to tensorboard
    if not opts.get("no_tensorboard", False):
        tb_logger.log_value("loss", loss, step)

        # Log representative predictions
        if output.dim() >= 1:
            representative_out = output[0].detach().cpu().numpy()
            for i, val in enumerate(representative_out):
                tb_logger.log_value(f"pred_step_{i}", val, step)

        tb_logger.log_value("grad_norm", grad_norms_all[0], step)
        tb_logger.log_value("grad_norm_clipped", grad_norms_clipped[0], step)

    # Log to WandB
    if opts.get("wandb_mode") != "disabled":
        wandb.log(
            {
                "train/loss": loss,
                "train/grad_norm": grad_norms_all[0],
                "train/grad_norm_clipped": grad_norms_clipped[0],
                "epoch": epoch,
                "step": step,
            }
        )


def log_epoch(
    epoch: int,
    epoch_duration: float,
    optimizer: torch.optim.Optimizer,
    opts: dict[str, Any],
    avg_loss: float | None = None,
) -> None:
    """
    Log summary of an epoch.
    """
    lr = optimizer.param_groups[0]["lr"]
    log_msg = f"Finished epoch {epoch}, took {time.strftime('%H:%M:%S', time.gmtime(epoch_duration))} s, lr={lr:.6f}"
    if avg_loss is not None:
        log_msg += f", avg_loss={avg_loss:.6f}"
    print(log_msg)

    if opts.get("wandb_mode") != "disabled":
        wandb_log = {"epoch/duration": epoch_duration, "epoch/lr": lr, "epoch": epoch}
        if avg_loss is not None:
            wandb_log["epoch/avg_loss"] = avg_loss
        wandb.log(wandb_log)


def plot_training_results(
    df: pd.DataFrame, save_path: str, title: str = "Training Progress"
) -> None:
    """
    Generate and save training progress plots.
    """
    plt.figure(figsize=(10, 6))
    if "loss" in df.columns:
        plt.plot(df["loss"], label="Loss")

    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def _convert_numpy(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: _convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy(v) for v in obj]
    return obj


def log_to_json_resilient(
    json_path: str, data: dict[str, Any], lock: Any | None = None
) -> None:
    """
    Thread-safe and error-resilient JSON logger using a temporary file fallback.
    """
    if lock is not None:
        lock.acquire()

    try:
        current_data = {}
        if os.path.isfile(json_path):
            try:
                with open(json_path) as f:
                    current_data = json.load(f)
            except (OSError, json.JSONDecodeError):
                print(f"Warning: Failed to read {json_path}, starting fresh.")

        # Merge data
        current_data.update(data)

        # Write with fallback
        try:
            with open(json_path, "w") as f:
                json.dump(_convert_numpy(current_data), f, indent=4)
        except Exception as e:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            fallback_path = f"{json_path}.{timestamp}.bak"
            print(f"Error writing to {json_path}: {e}. Falling back to {fallback_path}")
            with open(fallback_path, "w") as f:
                json.dump(_convert_numpy(current_data), f, indent=4)

    finally:
        if lock is not None:
            lock.release()
