"""
Visualization utilities for the routing problems.

This module provides functions for:
- Visualizing routing solutions and graphs.
- Plotting loss landscapes (if used).
- Creating PCA visualizations of embeddings.
- Interfacing with TensorBoard for visual logging.
"""

import argparse
import os

import loss_landscapes
import matplotlib
import numpy as np
import seaborn as sns
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torch.utils.tensorboard.writer import SummaryWriter

from logic.src.models.attention_model import AttentionModel
from logic.src.models.subnets.gat_encoder import GraphAttentionEncoder
from logic.src.pipeline.reinforcement_learning.policies.local_search import (
    vectorized_two_opt,
)
from logic.src.utils.functions import load_problem

"""
Visualization utilities for model analysis.

This module provides tools for:
- Plotting training trajectories (PCA of weights).
- Logging weight distributions and node embeddings to TensorBoard.
- Generating attention heatmaps.
- Visualizing loss landscapes (Imitation Loss, RL Cost).
"""

# --- UTILS ---


def get_batch(device, size=50, batch_size=32, temporal_horizon=0):
    """
    Generates a random batch of VRP-like data for visualization purposes.

    Args:
        device (torch.device): Device to creating tensors on.
        size (int, optional): Graph size. Defaults to 50.
        batch_size (int, optional): Batch size. Defaults to 32.
        temporal_horizon (int, optional): Temporal horizon for features. Defaults to 0.

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

    batch = {
        "depot": depot,
        "loc": loc,
        "dist": dist_tensor,
        "demand": waste,
        "waste": waste,
        "max_waste": max_waste,
    }

    # Add dummy temporal features if needed
    for i in range(1, temporal_horizon + 1):
        batch[f"fill{i}"] = torch.rand(batch_size, size, device=device)

    return batch


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
    files.sort(key=lambda x: (int(x.split("-")[1].split(".")[0]) if "-" in x and "epoch" in x else 0))

    weights = []
    epochs = []

    for f in files:
        path = os.path.join(checkpoint_dir, f)
        try:
            checkpoint = torch.load(path, map_location="cpu")
            state_dict = checkpoint.get("model", checkpoint)

            # More robust weight extraction: try encoder, then embed, then any model param
            weights_to_flat = []
            for k, p in state_dict.items():
                if "encoder" in k.lower() or "embed" in k.lower() or "model" in k.lower():
                    weights_to_flat.append(p.flatten())

            if not weights_to_flat:
                weights_to_flat = [p.flatten() for p in state_dict.values() if isinstance(p, torch.Tensor)]

            if not weights_to_flat:
                print(f"No valid tensors found in {f}")
                continue

            flat_weight = torch.cat(weights_to_flat).numpy()
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

        writer.add_embedding(sample_embeddings, metadata=labels, tag=f"Node_Embeddings_Ep{epoch}")

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


def plot_logit_lens(model, x_batch, output_file, epoch=0):
    """
    Implements the 'Logit Lens' technique for Attention Models.
    Project intermediate encoder layer outputs through the decoder's
    attention mechanism to see what the 'best' node is at each layer.

    Args:
        model (nn.Module): The Attention Model.
        x_batch (dict): Input batch.
        output_file (str): Filename to save the heatmap.
        epoch (int): Current epoch.
    """
    print("Computing Logit Lens...")
    model.eval()

    # Ensure compatible devices
    dev = next(model.parameters()).device
    x_batch = {k: v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in x_batch.items()}

    with torch.no_grad():
        h = model._get_initial_embeddings(x_batch)
        edges = x_batch.get("edges", None)

        all_probs = []

        # Helper to get first step probs
        def get_probs(embeddings):
            """Compute probability distribution for a given set of embeddings."""
            fixed = model.decoder._precompute(embeddings)
            # Correctly instantiate state with all required arguments
            state = model.problem.make_state(
                x_batch,
                edges=edges,
                cost_weights=getattr(model, "cost_weights", None),
                dist_matrix=x_batch.get("dist"),
            )
            log_p, _ = model.decoder._get_log_p(fixed, state)
            return log_p.exp()  # (Batch, 1, Nodes)

        # 0. Initial Embeddings (Layer 0)
        all_probs.append(get_probs(h))

        # 1. Intermediate Layers
        curr = h
        if hasattr(model.embedder, "layers"):
            for i, layer in enumerate(model.embedder.layers):
                curr = layer(curr, mask=edges)
                all_probs.append(get_probs(curr))

        # Final dropout/projection if any
        if hasattr(model.embedder, "dropout"):
            curr = model.embedder.dropout(curr)
            # Only add if it changed something or if we want to see the final output
            # Usually redundant if dropout is 0 during eval, but good for completeness
            # all_probs.append(get_probs(curr))

        # Shape: (Batch, NumLayers, Nodes)
        probs_tensor = torch.cat(all_probs, dim=1).cpu().numpy()

        # Visualize first sample in batch
        sample_probs = probs_tensor[0]  # (L, N)

        plt.figure(figsize=(12, 8))
        sns.heatmap(sample_probs, cmap="viridis", annot=False)
        plt.title(f"Logit Lens - Probability Distribution per Layer (Epoch {epoch})")
        plt.xlabel("Node Index")
        plt.ylabel("Encoder Layer Index")

        # Highlight top prediction per layer
        top_indices = np.argmax(sample_probs, axis=1)
        for layer_idx, node_idx in enumerate(top_indices):
            plt.text(
                node_idx + 0.5,
                layer_idx + 0.5,
                f"{node_idx}",
                color="white",
                ha="center",
                va="center",
                weight="bold",
                fontsize=8,
            )

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file)
        plt.close()

    print(f"Logit Lens saved to {output_file}")


# --- LOSS LANDSCAPE FUNCTIONS ---


def imitation_loss_fn(m, x_batch, pi_target, cost_weights=None):
    """
    Computes imitation loss (log likelihood of target) for loss landscape.
    """
    model_to_call = m.modules[0] if hasattr(m, "modules") else m
    if hasattr(model_to_call, "model"):
        model_to_call = model_to_call.model
    model_to_call.eval()

    # Ensure cost_weights are on the same device as the model
    dev = next(model_to_call.parameters()).device
    if cost_weights is not None:
        cost_weights = {k: v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in cost_weights.items()}
    elif hasattr(model_to_call, "cost_weights"):
        cost_weights = {
            k: v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in model_to_call.cost_weights.items()
        }

    with torch.no_grad():
        res = model_to_call(x_batch, cost_weights=cost_weights, return_pi=False, expert_pi=pi_target)
        log_likelihood = res[1]
    return -log_likelihood.mean().item()


def rl_loss_fn(m, x_batch, cost_weights=None):
    """
    Computes RL loss (greedy cost) for loss landscape.
    """
    model_to_call = m.modules[0] if hasattr(m, "modules") else m
    if hasattr(model_to_call, "model"):
        model_to_call = model_to_call.model
    model_to_call.eval()

    # Ensure cost_weights are on the same device as the model
    dev = next(model_to_call.parameters()).device
    if cost_weights is not None:
        cost_weights = {k: v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in cost_weights.items()}
    elif hasattr(model_to_call, "cost_weights"):
        cost_weights = {
            k: v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in model_to_call.cost_weights.items()
        }

    with torch.no_grad():
        model_to_call.set_decode_type("greedy")
        cost, _, _, _, _ = model_to_call(x_batch, cost_weights=cost_weights, return_pi=False)
    return cost.float().mean().item()


def plot_loss_landscape(model, opts, output_dir, epoch=0, size=50, batch_size=16, resolution=10, span=1.0):
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
    x_batch = get_batch(
        device,
        size=size,
        batch_size=batch_size,
        temporal_horizon=opts.get("temporal_horizon", 0),
    )

    print("Generating expert targets for landscape...")
    model.set_decode_type("greedy")
    with torch.no_grad():
        _, _, _, pi, _ = model(x_batch, return_pi=True)
        x_dist = x_batch["dist"]
        if x_dist.dim() == 2:
            x_dist = x_dist.unsqueeze(0)
        if x_dist.size(0) == 1:
            x_dist = x_dist.expand(pi.size(0), -1, -1)
        pi_with_depot = torch.cat([torch.zeros((pi.size(0), 1), dtype=torch.long, device=device), pi], dim=1)
        pi_opt = vectorized_two_opt(pi_with_depot, x_dist, max_iterations=100)
        pi_target = pi_opt[:, 1:]

    wrapped_model = MyModelWrapper(model)

    # Ensure all tensors are on the same device as the model
    # loss-landscapes might deepcopy the model, we want to be sure our batch matches
    model_device = next(model.parameters()).device

    # Helper to move dict of tensors to device
    def move_dict_to_device(d, dev):
        """Recursively move dictionary values to the specified device."""
        return {k: v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in d.items()}

    x_batch = move_dict_to_device(x_batch, model_device)
    pi_target = pi_target.to(model_device)

    # Imitation
    print(f"Computing Imitation Landscape on {model_device}...")

    # Store original cost weights to pass to metrics
    orig_cost_weights = getattr(model, "cost_weights", None)

    def imitation_metric(m):
        """Computes imitation loss for the current model state."""
        # Visualization is now CPU-only for robustness
        m_dev = torch.device("cpu")
        x_m = move_dict_to_device(x_batch, m_dev)
        pi_m = pi_target.to(m_dev)
        return imitation_loss_fn(m, x_m, pi_m, cost_weights=orig_cost_weights)

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
    print(f"Computing RL Cost Landscape on {model_device}...")

    def rl_metric(m):
        """Computes RL cost for the current model state."""
        m_dev = torch.device("cpu")
        x_m = move_dict_to_device(x_batch, m_dev)
        return rl_loss_fn(m, x_m, cost_weights=orig_cost_weights)

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
    """
    viz_modes = opts.get("viz_modes", [])
    if not viz_modes:
        return

    log_dir = opts.get("log_dir", "logs")
    viz_output_dir = os.path.join(log_dir, "visualizations")
    os.makedirs(viz_output_dir, exist_ok=True)

    print(f"\n--- Visualizing Epoch {epoch} ---")

    # Move model to CPU for visualization to avoid device mismatch issues with deepcopies/landscapes
    orig_device = next(model.parameters()).device
    model.cpu()

    # Temporarily update opts device
    viz_opts = opts.copy()
    viz_opts["device"] = torch.device("cpu")

    # Move cost weights to CPU if they exist
    if hasattr(model, "cost_weights"):
        model.cost_weights = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in model.cost_weights.items()}

    try:
        if "distributions" in viz_modes or "both" in viz_modes:
            writer = tb_logger.writer if tb_logger is not None and hasattr(tb_logger, "writer") else None
            log_weight_distributions(
                model,
                epoch,
                log_dir=os.path.join(log_dir, opts["run_name"]),
                writer=writer,
            )

        if "embeddings" in viz_modes:
            x_batch = get_batch(
                viz_opts["device"],
                size=opts["graph_size"],
                batch_size=1,
                temporal_horizon=opts.get("temporal_horizon", 0),
            )
            writer = tb_logger.writer if tb_logger is not None and hasattr(tb_logger, "writer") else None
            project_node_embeddings(
                model,
                x_batch,
                log_dir=os.path.join(log_dir, opts["run_name"]),
                writer=writer,
                epoch=epoch,
            )

        if "heatmaps" in viz_modes:
            plot_attention_heatmaps(model, viz_output_dir, epoch=epoch)

        if "logit_lens" in viz_modes:
            x_batch = get_batch(
                viz_opts["device"],
                size=opts["graph_size"],
                batch_size=1,
                temporal_horizon=opts.get("temporal_horizon", 0),
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
                size=opts["graph_size"],
                batch_size=4,
                resolution=10,
            )

        if "trajectory" in viz_modes:
            checkpoint_dir = opts["save_dir"]
            plot_weight_trajectories(checkpoint_dir, os.path.join(viz_output_dir, "trajectory.png"))

    finally:
        # Restore model to original device
        model.to(orig_device)
        if hasattr(model, "cost_weights"):
            model.cost_weights = {
                k: v.to(orig_device) if isinstance(v, torch.Tensor) else v for k, v in model.cost_weights.items()
            }

    print("Visualization complete.\n")


def main():
    """Main execution entry point for visualization debugging."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="Path to model checkpoint")
    parser.add_argument("--checkpoint_dir", type=str, help="Directory with multiple checkpoints")
    parser.add_argument("--output_dir", type=str, default="visualizations", help="Output directory")
    parser.add_argument("--log_dir", type=str, default="logs", help="TensorBoard log directory")

    parser.add_argument("--size", type=int, default=100, help="Problem size")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("--resolution", type=int, default=10, help="Resolution for landscapes")
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
            "logit_lens",
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
        plot_weight_trajectories(args.checkpoint_dir, os.path.join(args.output_dir, "trajectory.png"))
        return  # Trajectory doesn't need model loading

    # Load model
    if not args.model_path:
        raise ValueError("--model_path required for this mode")
    model = load_model_instance(args.model_path, device, size=args.size, problem_name=args.problem)

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

    elif args.mode == "logit_lens":
        x_batch = get_batch(device, size=args.size, batch_size=1)
        plot_logit_lens(model, x_batch, os.path.join(args.output_dir, "logit_lens.png"))

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
