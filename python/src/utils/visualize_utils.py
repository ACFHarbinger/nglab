"""
Visualization utilities for the routing problems.

This module provides functions for:
- Visualizing routing solutions and graphs.
- Plotting loss landscapes (if used).
- Creating PCA visualizations of embeddings.
- Interfacing with TensorBoard for visual logging.
"""

import os
import torch
import argparse
import numpy as np
import seaborn as sns
import loss_landscapes
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from torch.utils.tensorboard.writer import SummaryWriter
from logic.src.utils.functions import load_problem
from logic.src.models.attention_model import AttentionModel
from logic.src.models.subnets.gat_encoder import GraphAttentionEncoder
from logic.src.pipeline.reinforcement_learning.core.post_processing import (
    local_search_2opt_vectorized,
)


"""
Visualization utilities for model analysis.

This module provides tools for:
- Plotting training trajectories (PCA of weights).
- Logging weight distributions and node embeddings to TensorBoard.
- Generating attention heatmaps.
- Visualizing loss landscapes (Imitation Loss, RL Cost).
"""

# --- UTILS ---


def get_batch(device, size=50, batch_size=32):
    """
    Generates a random batch of VRP-like data for visualization purposes.

    Args:
        device (torch.device): Device to creating tensors on.
        size (int, optional): Graph size. Defaults to 50.
        batch_size (int, optional): Batch size. Defaults to 32.

    Returns:
        dict: Batch dictionary with keys 'depot', 'loc', 'dist', 'demand', etc.
    """
    # TODO: This should ideally use the problem's generate_instance or make_dataset
    all_coords = torch.rand(batch_size, size + 1, 2, device=device)
    depot = all_coords[:, 0, :]
    loc = all_coords[:, 1:, :]
    dist_tensor = torch.cdist(all_coords, all_coords)
    waste = torch.rand(batch_size, size, device=device)
    max_waste = torch.ones(batch_size, device=device)

    return {
        "depot": depot,
        "loc": loc,
        "dist": dist_tensor,
        "demand": waste,
        "waste": waste,
        "max_waste": max_waste,
    }


class MyModelWrapper(torch.nn.Module):
    """
    Wraps a model to conform to the interface expected by loss-landscapes library.
    """

    def __init__(self, model):
        """Initializes the wrapper."""
        super().__init__()
        self.model = model

    def forward(
        self,
        input,
        cost_weights=None,
        return_pi=False,
        pad=False,
        mask=None,
        expert_pi=None,
    ):
        """Forward pass of the model."""
        return self.model(input, cost_weights, return_pi, pad, mask, expert_pi)


def load_model_instance(model_path, device, size=100, problem_name="wcvrp"):
    """
    Loads a model for visualization, instantiating it with default architecture parameters.
    Note: Architecture parameters are currently hardcoded for visualization defaults.

    Args:
        model_path (str): Path to checkpoint.
        device (torch.device): Device.
        size (int, optional): Problem size. Defaults to 100.
        problem_name (str, optional): Problem name. Defaults to 'wcvrp'.

    Returns:
        nn.Module: Loaded model.
    """
    # This is a bit brittle as it assumes specific model args.
    # Ideally should load args from checkpoint or args.json
    problem = load_problem(problem_name)
    model = AttentionModel(
        embedding_dim=128,
        hidden_dim=512,
        problem=problem,
        encoder_class=GraphAttentionEncoder,
        n_encode_layers=3,
        mask_inner=True,
        mask_logits=True,
        normalization="instance",
        tanh_clipping=10.0,
        checkpoint_encoder=False,
        shrink_size=None,
        n_heads=8,
        n_encode_sublayers=1,
        n_decode_layers=2,
        predictor_layers=2,
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    # Handle cases where checkpoint might be nested
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model


# --- VISUALIZATION FUNCTIONS ---


def plot_weight_trajectories(checkpoint_dir, output_file):
    """
    Plots the trajectory of model weights across epochs using PCA projection.

    Args:
        checkpoint_dir (str): Directory containing checkpoints.
        output_file (str): Output image filename.
    """
    print("Computing Weight Trajectories...")
    # Create output dir if needed
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")]
    # Sort files by epoch number
    files.sort(
        key=lambda x: (
            int(x.split("-")[1].split(".")[0]) if "-" in x and "epoch" in x else 0
        )
    )

    weights = []
    epochs = []

    for f in files:
        path = os.path.join(checkpoint_dir, f)
        try:
            checkpoint = torch.load(path, map_location="cpu")
            state_dict = checkpoint.get("model", checkpoint)
            # Only take a subset of weights to avoid OOM or slow processing
            # e.g. just encoder or first layer
            flat_weight = torch.cat(
                [p.flatten() for k, p in state_dict.items() if "encoder" in k]
            ).numpy()
            if len(flat_weight) == 0:  # Fallback if no encoder
                flat_weight = torch.cat(
                    [p.flatten() for p in state_dict.values()]
                ).numpy()
            weights.append(flat_weight)
            epochs.append(f.replace(".pt", ""))
        except Exception as e:
            print(f"Skipping {f} due to error: {e}")

    if not weights:
        print("No weights found.")
        return

    weights = np.array(weights)
    if weights.shape[0] < 2:
        print("Not enough checkpoints for trajectory.")
        return

    pca = PCA(n_components=2)
    projected = pca.fit_transform(weights)

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


def log_weight_distributions(model, epoch, log_dir, writer=None):
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


def project_node_embeddings(model, x_batch, log_dir, writer=None, epoch=0):
    """
    Projects node embeddings to 3D space in TensorBoard Projector.

    Args:
        model (nn.Module): The model.
        x_batch (dict): Input batch.
        log_dir (str): TensorBoard log directory (used if writer is None).
        writer (SummaryWriter, optional): Existing writer.
        epoch (int, optional): Current epoch. Defaults to 0.
    """
    close_writer = False
    if writer is None:
        print(f"Projecting Node Embeddings to TensorBoard at {log_dir}...")
        writer = SummaryWriter(log_dir)
        close_writer = True

    model.eval()
    with torch.no_grad():
        # Get initial node embeddings
        nodes = model._get_initial_embeddings(x_batch)
        edges = x_batch.get("edges", None)
        dist_matrix = x_batch.get("dist", None)

        # Check if embedder accepts dist (e.g. GAT with edge embeddings)
        if getattr(model.embedder, "init_edge_embed", None) is not None:
            embeddings = model.embedder(nodes, edges, dist=dist_matrix)
        else:
            embeddings = model.embedder(nodes, edges)

        # Take the first sample in batch
        sample_embeddings = embeddings[0]  # (G, D)
        labels = [f"Node_{i}" for i in range(sample_embeddings.size(0))]
        labels[0] = "Depot"

        writer.add_embedding(
            sample_embeddings, metadata=labels, tag=f"Node_Embeddings_Ep{epoch}"
        )

    if close_writer:
        writer.close()
        print("Embeddings projected.")


def plot_attention_heatmaps(model, output_dir, epoch=0):
    """
    Plots heatmaps of attention weights (Query, Key, Value) for all layers.

    Args:
        model (nn.Module): The model.
        output_dir (str): Directory to save images.
        epoch (int, optional): Current epoch. Defaults to 0.
    """
    print("Plotting Attention Heatmaps...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model.eval()
    # Encoder attention weights
    # Structure: model.embedder.layers[i].att.module.W_query.weight etc.
    # Looking at gat_encoder.py: self.layers[i].att.module is MultiHeadAttention

    for i, layer in enumerate(model.embedder.layers):
        # att is SkipConnection, att.module is MultiHeadAttention
        mha = layer.att.module

        # W_query, W_key, W_val are typically the names in MultiHeadAttention
        for name in ["W_query", "W_key", "W_val"]:
            if hasattr(mha, name):
                param = getattr(mha, name)
                if hasattr(param, "weight"):
                    weight = param.weight.data.cpu().numpy()
                else:
                    weight = param.data.cpu().numpy()

                # Handle 3D weights (Multi-Head Attention)
                if weight.ndim == 3:
                    # (Heads, Input, Output) -> (Input, Heads * Output)
                    h, input_dim, o = weight.shape
                    weight = weight.transpose(1, 0, 2).reshape(input_dim, h * o)
                elif weight.ndim > 2:
                    # Generic flatten to 2D
                    weight = weight.reshape(weight.shape[0], -1)

                plt.figure(figsize=(12, 10))
                sns.heatmap(weight, cmap="RdBu_r", center=0)
                plt.title(f"Layer {i} - {name} Weights (Epoch {epoch})")
                plt.savefig(os.path.join(output_dir, f"layer_{i}_{name}_ep{epoch}.png"))
                plt.close()

    print(f"Heatmaps saved to {output_dir}")


# --- LOSS LANDSCAPE FUNCTIONS ---


def imitation_loss_fn(m, x_batch, pi_target):
    """
    Computes imitation loss (log likelihood of target) for loss landscape.
    Note: Returns negation because loss landscape plots typically visualize 'loss' where lower is better/blue.
    Wait, actually landscapes visualize Z. If we want to visualize Likelihood, we might want negative log likelihood (NLL) as loss.
    Here we return -log_likelihood, which is NLL (Loss).

    Args:
        m (nn.Module): Wrapped model.
        x_batch (dict): Input batch.
        pi_target (Tensor): Target tour.

    Returns:
        float: Negative Log Likelihood.
    """
    model_to_call = m.modules[0] if hasattr(m, "modules") else m
    if hasattr(model_to_call, "model"):
        model_to_call = model_to_call.model
    model_to_call.eval()
    with torch.no_grad():
        res = model_to_call(x_batch, return_pi=False, expert_pi=pi_target)
        log_likelihood = res[1]
    return -log_likelihood.mean().item()


def rl_loss_fn(m, x_batch):
    """
    Computes RL loss (greedy cost) for loss landscape.

    Args:
        m (nn.Module): Wrapped model.
        x_batch (dict): Input batch.

    Returns:
        float: Mean cost.
    """
    model_to_call = m.modules[0] if hasattr(m, "modules") else m
    if hasattr(model_to_call, "model"):
        model_to_call = model_to_call.model
    model_to_call.eval()
    with torch.no_grad():
        model_to_call.set_decode_type("greedy")
        cost, _, _, _, _ = model_to_call(x_batch, return_pi=False)
    return cost.float().mean().item()


def plot_loss_landscape(
    model, opts, output_dir, epoch=0, size=50, batch_size=16, resolution=10, span=1.0
):
    """
    Computes and plots 2D and 3D loss landscapes for both Imitation Loss and RL Cost.

    Args:
        model (nn.Module): The model.
        opts (dict): Options containing 'device'.
        output_dir (str): Directory to save plots.
        epoch (int, optional): Current epoch.
        size (int, optional): Graph size. Defaults to 50.
        batch_size (int, optional): Batch size. Defaults to 16.
        resolution (int, optional): Grid resolution. Defaults to 10.
        span (float, optional): Range of perturbation. Defaults to 1.0.
    """
    print("Computing Loss Landscape...")
    os.makedirs(output_dir, exist_ok=True)
    device = opts["device"]

    # Generate random batch for landscape
    # TODO: Use problem.make_dataset if possible for consistency
    x_batch = get_batch(device, size=size, batch_size=batch_size)

    print("Generating expert targets for landscape...")
    model.set_decode_type("greedy")
    with torch.no_grad():
        _, _, _, pi, _ = model(x_batch, return_pi=True)
        x_dist = x_batch["dist"]
        if x_dist.dim() == 2:
            x_dist = x_dist.unsqueeze(0)
        if x_dist.size(0) == 1:
            x_dist = x_dist.expand(pi.size(0), -1, -1)
        pi_with_depot = torch.cat(
            [torch.zeros((pi.size(0), 1), dtype=torch.long, device=device), pi], dim=1
        )
        pi_opt = local_search_2opt_vectorized(pi_with_depot, x_dist, max_iterations=100)
        pi_target = pi_opt[:, 1:]

    wrapped_model = MyModelWrapper(model)

    # Imitation
    print("Computing Imitation Landscape...")

    def imitation_metric(m):
        """Computes imitation loss for the current model state."""
        return imitation_loss_fn(m, x_batch, pi_target)

    try:
        data = loss_landscapes.random_plane(
            wrapped_model,
            imitation_metric,
            distance=span,
            steps=resolution,
            deepcopy_model=True,
        )

        plt.figure()
        plt.contour(data, levels=50)
        plt.title(f"Imitation Loss Landscape (Epoch {epoch})")
        plt.savefig(os.path.join(output_dir, f"landscape_imitation_ep{epoch}.png"))
        plt.close()

        plt.close()

        plt.figure()
        ax = plt.axes(projection="3d")
        X, Y = np.meshgrid(np.arange(resolution), np.arange(resolution))
        ax.plot_surface(X, Y, np.array(data), cmap="viridis")
        plt.title(f"Imitation Loss Surface (Epoch {epoch})")
        plt.savefig(os.path.join(output_dir, f"surface_imitation_ep{epoch}.png"))
        plt.close()
    except Exception as e:
        print(f"Error computing imitation landscape: {e}")

    # RL
    print("Computing RL Cost Landscape...")

    def rl_metric(m):
        """Computes RL cost for the current model state."""
        return rl_loss_fn(m, x_batch)

    try:
        data = loss_landscapes.random_plane(
            wrapped_model,
            rl_metric,
            distance=span,
            steps=resolution,
            deepcopy_model=True,
        )

        plt.figure()
        plt.contour(data, levels=50)
        plt.title(f"RL Cost Landscape (Epoch {epoch})")
        plt.savefig(os.path.join(output_dir, f"landscape_rl_ep{epoch}.png"))
        plt.close()

        plt.close()

        plt.figure()
        ax = plt.axes(projection="3d")
        X, Y = np.meshgrid(np.arange(resolution), np.arange(resolution))
        ax.plot_surface(X, Y, np.array(data), cmap="magma")
        plt.title(f"RL Cost Surface (Epoch {epoch})")
        plt.savefig(os.path.join(output_dir, f"surface_rl_ep{epoch}.png"))
        plt.close()
    except Exception as e:
        print(f"Error computing RL landscape: {e}")


# --- MAIN CONTROLLER ---


def visualize_epoch(model, problem, opts, epoch, tb_logger=None):
    """
    Main entry point for visualization during training.
    Executes selected visualization modes (distributions, embeddings, loss, etc.).

    Args:
        model (nn.Module): The model.
        problem (class): Problem class (unused currently but kept for interface).
        opts (dict): Options dictionary.
        epoch (int): Current epoch number.
        tb_logger (Logger, optional): TensorBoard logger instance.
    """
    viz_modes = opts.get("viz_modes", [])
    log_dir = opts.get("log_dir", "logs")
    # If using tensorboard logger class, get the log_dir from it if possible
    if tb_logger is not None and hasattr(tb_logger, "log_dir"):
        # tb_logger is likely logic.src.utils.log_utils.Logger or similar wrapper
        # Assuming it writes to some dir, or we just rely on opts['log_dir'] which is usually base
        pass

    viz_output_dir = os.path.join(opts["output_dir"], "visualizations")
    os.makedirs(viz_output_dir, exist_ok=True)

    print(f"\n--- Visualizing Epoch {epoch} ---")

    if "distributions" in viz_modes or "both" in viz_modes:
        # Pass tb_logger writer if available, else will create one
        writer = (
            tb_logger.writer
            if tb_logger is not None and hasattr(tb_logger, "writer")
            else None
        )
        log_weight_distributions(
            model, epoch, log_dir=os.path.join(log_dir, opts["run_name"]), writer=writer
        )

    if "embeddings" in viz_modes:
        x_batch = get_batch(opts["device"], size=opts["graph_size"], batch_size=1)
        writer = (
            tb_logger.writer
            if tb_logger is not None and hasattr(tb_logger, "writer")
            else None
        )
        project_node_embeddings(
            model,
            x_batch,
            log_dir=os.path.join(log_dir, opts["run_name"]),
            writer=writer,
            epoch=epoch,
        )

    if "heatmaps" in viz_modes:
        plot_attention_heatmaps(model, viz_output_dir, epoch=epoch)

    if "loss" in viz_modes or "both" in viz_modes:
        plot_loss_landscape(
            model,
            opts,
            viz_output_dir,
            epoch=epoch,
            size=opts["graph_size"],
            batch_size=4,
            resolution=10,
        )  # Keep batch/res low for speed

    if "trajectory" in viz_modes:
        # Trajectory needs history of checkpoints, so we pass the checkpoint dir
        checkpoint_dir = opts["save_dir"]
        plot_weight_trajectories(
            checkpoint_dir, os.path.join(viz_output_dir, "trajectory.png")
        )

    print("Visualization complete.\n")


def main():
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
        "--log_dir", type=str, default="runs/viz", help="TensorBoard log directory"
    )

    parser.add_argument("--size", type=int, default=100, help="Problem size")
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--resolution", type=int, default=10, help="Resolution for landscapes"
    )
    parser.add_argument("--span", type=float, default=1.0, help="Span for landscapes")
    parser.add_argument("--problem", type=str, default="wcvrp", help="Problem type")

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=[
            "trajectory",
            "distributions",
            "embeddings",
            "heatmaps",
            "loss",
            "both",
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
        return  # Trajectory doesn't need model loading

    # Load model
    if not args.model_path:
        raise ValueError("--model_path required for this mode")
    model = load_model_instance(
        args.model_path, device, size=args.size, problem_name=args.problem
    )

    if args.mode == "distributions":
        if not args.checkpoint_dir:
            # If no dir, just log current?
            log_weight_distributions(model, 0, args.log_dir)
        else:
            # If checkpoint dir provided, maybe iterate? But main logic above assumes single model.
            # For standalone, maybe we just do the one.
            log_weight_distributions(model, 0, args.log_dir)

    elif args.mode == "embeddings":
        x_batch = get_batch(device, size=args.size, batch_size=1)
        project_node_embeddings(model, x_batch, args.log_dir)

    elif args.mode == "heatmaps":
        plot_attention_heatmaps(model, args.output_dir)

    elif args.mode == "loss" or args.mode == "both":
        fake_opts = {"device": device}
        plot_loss_landscape(
            model,
            fake_opts,
            args.output_dir,
            size=args.size,
            batch_size=args.batch_size,
            resolution=args.resolution,
            span=args.span,
        )


if __name__ == "__main__":
    main()
