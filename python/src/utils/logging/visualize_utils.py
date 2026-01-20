"""
Visualization utilities for model analysis and the NGLab trading environment.

This module provides tools for:
- Plotting training trajectories (PCA of weights).
- Logging weight distributions to TensorBoard.
- Implementing the 'Logit Lens' for NSTransformer models.
"""

import argparse
import os
from typing import Any

import loss_landscapes
import matplotlib
import numpy as np
import seaborn as sns
import torch
from torch import nn

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torch.utils.tensorboard.writer import SummaryWriter

from python.src.utils.functions.functions import load_model

# --- UTILS ---


def get_batch(
    device: torch.device, seq_len: int = 30, batch_size: int = 32, feature_dim: int = 12
) -> torch.Tensor:
    """
    Generates a random batch of sequential trading data for visualization purposes.

    Args:
        device (torch.device): Device to create tensors on.
        seq_len (int): Sequence length (lookback).
        batch_size (int): Batch size.
        feature_dim (int): Number of features per step.

    Returns:
        torch.Tensor: Batch tensor [Batch, SeqLen, FeatureDim].
    """
    return torch.randn(batch_size, seq_len, feature_dim, device=device)


class MyModelWrapper(nn.Module):
    """
    Wraps a model to conform to the interface expected by loss-landscapes library.
    """

    def __init__(self, model: nn.Module):
        """Initializes the wrapper."""
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        return self.model(x)


def load_model_instance(
    model_path: str, device: torch.device
) -> tuple[nn.Module, dict[str, Any]]:
    """
    Loads a model for visualization using the project's standard loading utility.

    Args:
        model_path (str): Path to checkpoint or directory.
        device (torch.device): Device.

    Returns:
        Tuple[nn.Module, Dict[str, Any]]: Loaded model and options.
    """
    model, opts = load_model(model_path)
    model.to(device)
    model.eval()
    return model, opts


# --- VISUALIZATION FUNCTIONS ---


def plot_weight_trajectories(checkpoint_dir: str, output_file: str) -> None:
    """
    Plots the trajectory of model weights across epochs using PCA projection.

    Args:
        checkpoint_dir (str): Directory containing checkpoints.
        output_file (str): Output image filename.
    """
    print("Computing Weight Trajectories...")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")]
    # Sort files by epoch number if possible
    try:
        files.sort(
            key=lambda x: (
                int(x.split("-")[1].split(".")[0]) if "-" in x and "epoch" in x else 0
            )
        )
    except (IndexError, ValueError):
        files.sort()

    weights = []
    epochs = []

    for f in files:
        path = os.path.join(checkpoint_dir, f)
        try:
            checkpoint = torch.load(path, map_location="cpu")
            # Handle standard checkpoint format where 'model' key stores state_dict
            state_dict = checkpoint.get("model", checkpoint)

            # Flatten all parameters into a single vector
            tensors = [
                p.flatten() for p in state_dict.values() if isinstance(p, torch.Tensor)
            ]
            if not tensors:
                continue
            flat_weight = torch.cat(tensors).numpy()

            weights.append(flat_weight)
            epochs.append(f.replace(".pt", ""))
        except Exception as e:
            print(f"Skipping {f} due to error: {e}")

    if len(weights) < 2:
        print("Not enough valid checkpoints for trajectory.")
        return

    weights_arr = np.array(weights)
    pca = PCA(n_components=2)
    projected = pca.fit_transform(weights_arr)

    plt.figure(figsize=(10, 8))
    plt.plot(projected[:, 0], projected[:, 1], "-o", alpha=0.6)
    for i, txt in enumerate(epochs):
        plt.annotate(txt, (projected[i, 0], projected[i, 1]), size=8)

    plt.title("Training Trajectory (PCA Projection of Model Weights)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)
    plt.savefig(output_file)
    plt.close()
    print(f"Trajectory saved to {output_file}")


def log_weight_distributions(
    model: nn.Module, epoch: int, log_dir: str, writer: SummaryWriter | None = None
) -> None:
    """
    Logs histograms of model weight distributions to TensorBoard.

    Args:
        model (nn.Module): The model.
        epoch (int): Current epoch.
        log_dir (str): TensorBoard log directory (used if writer is None).
        writer (SummaryWriter, optional): Existing writer.
    """
    close_writer = False
    if writer is None:
        print(f"Logging Weight Distributions to TensorBoard at {log_dir}...")
        writer = SummaryWriter(log_dir)
        close_writer = True

    for name, param in model.named_parameters():
        writer.add_histogram(name, param, epoch)

    if close_writer:
        writer.close()
        print("Distributions logged.")


def plot_logit_lens(
    model: nn.Module, x_batch: torch.Tensor, output_file: str, epoch: int = 0
) -> None:
    """
    Implements the 'Logit Lens' technique for NSTransformer models.
    Projects intermediate encoder layer outputs through the final projection
    layer to see how the prediction evolves.

    Args:
        model (nn.Module): The NSTransformer model.
        x_batch (torch.Tensor): Input batch [Batch, SeqLen, FeatureDim].
        output_file (str): Filename to save the heatmap.
        epoch (int): Current epoch.
    """
    print("Computing Logit Lens...")
    model.eval()
    dev = next(model.parameters()).device
    x_batch = x_batch.to(dev)

    if not hasattr(model, "encoder") or not hasattr(model, "projection"):
        print("Model does not support Logit Lens (missing encoder/projection).")
        return

    with torch.no_grad():
        # Dummy marks if models requires them
        x_mark = torch.zeros(x_batch.size(0), x_batch.size(1), 1, device=dev)

        try:
            # We follow the forward pass logic
            if hasattr(model, "enc_embedding"):
                curr = model.enc_embedding(x_batch, x_mark)
            else:
                curr = x_batch

            layer_outputs = []
            layer_outputs.append(curr)  # Layer 0: Embedding

            # Encoder layers
            encoder = model.encoder
            if hasattr(encoder, "attn_layers"):
                for attn_layer in encoder.attn_layers:
                    curr, _ = attn_layer(curr)
                    layer_outputs.append(curr)

            # Final projection of each layer output
            projection = model.projection
            preds = []
            for out in layer_outputs:
                p = projection(out)
                # Take last step of the sequence for visualization
                preds.append(p[0, -1, :].cpu().numpy())

            preds_arr = np.array(preds)

            plt.figure(figsize=(12, 6))
            sns.heatmap(preds_arr, cmap="RdYlGn", center=0, annot=True, fmt=".2f")
            plt.title(f"Logit Lens - Prediction Evolution (Epoch {epoch})")
            plt.xlabel("Output Dimension")
            plt.ylabel("Layer Index (0=Embed, 1..N=Encoder)")

            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            plt.savefig(output_file)
            plt.close()
            print(f"Logit Lens saved to {output_file}")

        except Exception as e:
            print(f"Failed to compute Logit Lens: {e}")


def mae_loss_fn(m: Any, x_batch: torch.Tensor, target: torch.Tensor) -> float:
    """
    Computes MAE loss for loss landscape.
    """
    # loss-landscapes wraps the model, we use .eval() if it supports it
    if hasattr(m, "eval"):
        m.eval()
    with torch.no_grad():
        # Use .forward() or __call__
        if hasattr(m, "forward"):
            output = m.forward(x_batch)
        else:
            output = m(x_batch)
        loss = torch.abs(output - target).mean()
    return float(loss.item())


def plot_loss_landscape(
    model: nn.Module,
    opts: dict[str, Any],
    output_dir: str,
    epoch: int = 0,
    seq_len: int = 30,
    batch_size: int = 16,
    resolution: int = 10,
    span: float = 1.0,
) -> None:
    """
    Computes and plots 2D loss landscapes for MAE loss.

    Args:
        model (nn.Module): The model.
        opts (dict): Options.
        output_dir (str): Directory to save plots.
        epoch (int): Current epoch.
        seq_len (int): Sequence length.
        batch_size (int): Batch size.
        resolution (int): Grid resolution.
        span (float): Range of perturbation.
    """
    print("Computing Loss Landscape...")
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(opts.get("device", "cpu"))

    x_batch = get_batch(device, seq_len=seq_len, batch_size=batch_size)

    with torch.no_grad():
        target = model(x_batch)

    print(f"Generating landscape surface ({resolution}x{resolution} grid)...")

    try:
        landscape = loss_landscapes.random_plane(
            model,
            lambda m: mae_loss_fn(m, x_batch, target),  # type: ignore[arg-type]
            distance=(
                int(span) if span >= 1 else 1
            ),  # random_plane distance usually takes int steps? Wait, Mypy said int.
            steps=resolution,
            normalization="filter",
            deepcopy_model=True,
        )

        plt.figure(figsize=(10, 8))
        plt.contourf(landscape, levels=20)
        plt.colorbar(label="MAE Loss")
        plt.title(f"Loss Landscape Contour (Epoch {epoch})")
        plt.savefig(os.path.join(output_dir, f"landscape_2d_epoch_{epoch}.png"))
        plt.close()

        print(f"Loss landscape saved to {output_dir}")
    except Exception as e:
        print(f"Failed to compute Loss Landscape: {e}")


# --- MAIN CONTROLLER ---


def visualize_epoch(
    model: nn.Module, opts: dict[str, Any], epoch: int, tb_logger: Any | None = None
) -> None:
    """
    Main entry point for visualization during training.
    """
    viz_modes = opts.get("viz_modes", [])
    if not viz_modes:
        return

    log_dir = opts.get("log_dir", "logs")
    viz_output_dir = os.path.join(log_dir, "visualizations")
    os.makedirs(viz_output_dir, exist_ok=True)

    print(f"\n--- Visualizing Epoch {epoch} ---")

    orig_device = next(model.parameters()).device
    model.cpu()

    viz_opts = opts.copy()
    viz_opts["device"] = torch.device("cpu")

    try:
        if "distributions" in viz_modes or "both" in viz_modes:
            writer = (
                tb_logger.writer
                if tb_logger is not None and hasattr(tb_logger, "writer")
                else None
            )
            log_weight_distributions(
                model,
                epoch,
                log_dir=os.path.join(log_dir, opts.get("run_name", "default")),
                writer=writer,
            )

        if "logit_lens" in viz_modes:
            x_batch = get_batch(
                viz_opts["device"], seq_len=opts.get("lookback", 30), batch_size=1
            )
            plot_logit_lens(
                model,
                x_batch,
                os.path.join(viz_output_dir, f"logit_lens_ep{epoch}.png"),
                epoch=epoch,
            )

        if "loss" in viz_modes or "both" in viz_modes:
            plot_loss_landscape(
                model,
                viz_opts,
                viz_output_dir,
                epoch=epoch,
                seq_len=opts.get("lookback", 30),
                batch_size=4,
                resolution=10,
            )

        if "trajectory" in viz_modes:
            checkpoint_dir = opts.get("save_dir", "checkpoints")
            plot_weight_trajectories(
                checkpoint_dir, os.path.join(viz_output_dir, "trajectory.png")
            )

    finally:
        model.to(orig_device)

    print("Visualization complete.\n")


def main() -> None:
    """Main execution entry point for visualization debugging."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="Path to model checkpoint")
    parser.add_argument(
        "--checkpoint_dir", type=str, help="Directory with multiple checkpoints"
    )
    parser.add_argument(
        "--output_dir", type=str, default="visualizations", help="Output directory"
    )
    parser.add_argument(
        "--log_dir", type=str, default="logs", help="TensorBoard log directory"
    )

    parser.add_argument("--seq_len", type=int, default=30, help="Sequence length")
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--resolution", type=int, default=10, help="Resolution for landscapes"
    )
    parser.add_argument("--span", type=float, default=1.0, help="Span for landscapes")

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=[
            "trajectory",
            "distributions",
            "logit_lens",
            "loss",
        ],
        help="Visualization mode",
    )

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == "trajectory":
        if not args.checkpoint_dir:
            raise ValueError("--checkpoint_dir required for trajectory")
        plot_weight_trajectories(
            args.checkpoint_dir, os.path.join(args.output_dir, "trajectory.png")
        )
        return

    if not args.model_path:
        raise ValueError("--model_path required for this mode")
    model, _ = load_model_instance(args.model_path, device)

    if args.mode == "distributions":
        log_weight_distributions(model, 0, args.log_dir)

    elif args.mode == "logit_lens":
        x_batch = get_batch(device, seq_len=args.seq_len, batch_size=1)
        plot_logit_lens(model, x_batch, os.path.join(args.output_dir, "logit_lens.png"))

    elif args.mode == "loss":
        fake_opts = {"device": device}
        plot_loss_landscape(
            model,
            fake_opts,
            args.output_dir,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            resolution=args.resolution,
            span=args.span,
        )


if __name__ == "__main__":
    main()
