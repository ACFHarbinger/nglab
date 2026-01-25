"""
Plotting utilities for the NGLab framework.

This module provides functions for generating various plots:
- Static line charts (training curves, Pareto fronts).
- TSP and VRP route visualizations using Matplotlib.
- Attention map heatmaps.
- Interactive visualizations.
"""

import math
import os
from collections.abc import Callable
from typing import Any

import numpy as np
import plotly.express as px
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from ..io.file_utils import compose_dirpath


def plot_linechart(  # noqa: PLR0913, PLR0915
    output_dest: str,
    graph_log: NDArray[Any],
    plot_func: Callable[..., Any],
    policies: list[str],
    x_label: str | None = None,
    y_label: str | None = None,
    title: str | None = None,
    fsave: bool = True,
    scale: str = "linear",
    x_values: list[float] | None = None,
    linestyles: list[str] | None = None,
    markers: list[str] | None = None,
    annotate: bool = True,
    pareto_front: bool = False,
) -> list[list[int]] | None:
    """
    Plots a generic line chart, optionally handling multiple policies and Pareto fronts.

    Returns:
        list or None: Pareto dominants if pareto_front is True, else None.
    """

    def plot_graphs_out(
        plot_func: Callable[..., Any],
        graph_log: NDArray[Any],
        x_values: list[float] | None,
        linestyles: list[str] | None,
        markers: list[str] | None,
    ) -> dict[int, list[tuple[float, float]]]:
        """
        Helper to plot graphs for different policies.
        """
        points_by_nbins: dict[int, list[tuple[float, float]]] = {}
        for idx, lg in enumerate(zip(*graph_log, strict=False)):
            if x_values is None:
                to_plot = (*lg,)
            else:
                to_plot = (
                    x_values,
                    *lg,
                )

            line: str | None = (
                linestyles[idx % len(linestyles)] if linestyles is not None else None
            )
            mark: str | None = (
                markers[idx % len(markers)] if markers is not None else None
            )

            if not line and not mark:
                plot_func(*to_plot)
            elif line and not mark:
                plot_func(*to_plot, linestyle=line)
            elif mark and not line:
                plot_func(*to_plot, marker=mark)
            else:
                plot_func(*to_plot, linestyle=line, marker=mark)

            # Extract points for Pareto
            # Assuming lg structure allows this indexing
            inner_lg = list(zip(*lg, strict=False))
            for i in range(len(inner_lg[0])):
                x = inner_lg[0][i]
                y = inner_lg[5][i]
                if idx not in points_by_nbins:
                    points_by_nbins[idx] = []
                points_by_nbins[idx].append((float(x), float(y)))
        return points_by_nbins

    plt.figure(dpi=200)
    if title is not None:
        plt.title(title)

    points_by_nbins: dict[int, list[tuple[float, float]]] = {}

    if len(graph_log.shape) == 2:
        points_by_nbins = {0: []}
        for idx, ll in enumerate(graph_log):
            if markers is not None:
                mark = markers[idx % len(markers)]
                plot_func(ll, mark)
            else:
                plot_func(ll)
            points_by_nbins[0].append((float(ll[0]), float(ll[5])))
    elif len(graph_log.shape) == 3:
        points_by_nbins = plot_graphs_out(
            plot_func, graph_log, x_values, linestyles, markers
        )
        if annotate:
            for lg in zip(*graph_log, strict=False):
                inner_lg = list(zip(*lg, strict=False))
                for i in range(len(inner_lg[0])):
                    x = inner_lg[0][i]
                    y = inner_lg[5][i]
                    if i == graph_log.shape[0] - 1:
                        plt.scatter(
                            x,
                            y,
                            s=200,
                            marker="o",
                            facecolors="none",
                            edgecolors="black",
                            linewidths=1,
                            zorder=10,
                        )

    pareto_dominants: list[list[int]] = []
    if pareto_front:
        for id_nbins, points in points_by_nbins.items():
            dominance_ls = [0] * len(points)
            pareto_points = []
            for id_point, point in enumerate(points):
                dominated = False
                for other_point in points:
                    if (
                        other_point[0] <= point[0]
                        and other_point[1] >= point[1]
                        and (other_point[0] < point[0] or other_point[1] > point[1])
                    ):
                        dominated = True
                        break
                if not dominated:
                    dominance_ls[id_point] = 1
                    pareto_points.append(point)

            pareto_dominants.append(dominance_ls)
            pareto_points.sort(key=lambda p: p[0])
            pareto_x = [p[0] for p in pareto_points]
            pareto_y = [p[1] for p in pareto_points]

            label = f"Pareto Front (ID {id_nbins})"
            plt.plot(
                pareto_x,
                pareto_y,
                "--",
                color="black",
                linewidth=2,
                label=label,
                zorder=5,
            )

    if scale != "linear":
        plt.yscale(scale)
        plt.xscale(scale)
    if x_label is not None:
        plt.xlabel(x_label)
    if y_label is not None:
        plt.ylabel(y_label)

    plt.legend(policies)
    if fsave:
        if x_values is not None:
            plt.savefig(output_dest)
        else:
            plt.savefig(output_dest, bbox_inches="tight")
    plt.show()
    return pareto_dominants if pareto_front else None


def discrete_cmap(n: int, base_cmap: str | Any | None = None) -> Any:
    """
    Create an N-bin discrete colormap from the specified input map.
    """
    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, n))
    cmap_name = base.name + str(n)
    return base.from_list(cmap_name, color_list, n)


@compose_dirpath
def plot_attention_maps_wrapper(  # noqa: PLR0913
    dir_path: str,
    attention_dict: dict[str, list[dict[str, torch.Tensor]]],
    model_name: str,
    execution_function: Callable[..., Any],
    layer_idx: int = 0,
    sample_idx: int = 0,
    head_idx: int = 0,
    batch_idx: int = 0,
    x_labels: list[str] | None = None,
    y_labels: list[str] | None = None,
    **execution_kwargs: Any,
) -> NDArray[Any]:
    """
    Plot attention maps as heatmaps for a given layer, head, batch, and simulation sample.
    """
    assert sample_idx >= 0, f"sample_idx {sample_idx} must be a non-negative integer"

    attention_weights = attention_dict[model_name][sample_idx]["attention_weights"]

    assert layer_idx < attention_weights.shape[0]
    assert head_idx < attention_weights.shape[1]
    assert batch_idx < attention_weights.shape[2]

    # Extract attention map
    if head_idx >= 0:
        if batch_idx >= 0:
            attn_map = attention_weights[layer_idx, head_idx, batch_idx].cpu().numpy()
            title = (
                f"Attention Map (Layer {layer_idx}, Head {head_idx}, Batch {batch_idx})"
            )
            attention_filename = os.path.join(
                dir_path,
                "attention_maps",
                model_name,
                f"layer{layer_idx}_head{head_idx}_map{sample_idx}.png",
            )
        else:
            attn_map = (
                attention_weights[layer_idx, head_idx, :].mean(dim=0).cpu().numpy()
            )  # Average over batches
            title = f"Attention Map Average Over All Batches (Layer {layer_idx}, Head {head_idx})"
            attention_filename = os.path.join(
                dir_path,
                "attention_maps",
                model_name,
                f"layer{layer_idx}_head{head_idx}_map{sample_idx}.png",
            )
    elif batch_idx >= 0:
        attn_map = (
            attention_weights[layer_idx, :, batch_idx].mean(dim=0).cpu().numpy()
        )  # Average over heads
        title = f"Attention Map Average Over All Heads (Layer {layer_idx}, Batch {batch_idx})"
        attention_filename = os.path.join(
            dir_path,
            "attention_maps",
            model_name,
            f"layer{layer_idx}_headavg_map{sample_idx}.png",
        )
    else:
        attn_map = (
            attention_weights[layer_idx, :, :].mean(dim=(0, 1)).cpu().numpy()
        )  # Average over heads and batches
        title = f"Attention Map Average Over All Heads and Batches (Layer {layer_idx})"
        attention_filename = os.path.join(
            dir_path,
            "attention_maps",
            model_name,
            f"layer{layer_idx}_headavg_map{sample_idx}.png",
        )

    os.makedirs(os.path.dirname(attention_filename), exist_ok=True)

    # Dynamically set figure size based on map_size
    base_vertexsize = 0.5
    map_size = math.isqrt(attn_map.shape[0] * attn_map.shape[1])
    min_figsize = 6.0
    max_figsize = 30.0
    figsize = min(max(min_figsize, base_vertexsize * map_size), max_figsize)
    fig = plt.figure(figsize=(figsize, figsize))

    # Adjust annotations and font sizes
    max_ticsize = 8
    max_annotsize = 8
    annot = map_size <= 55
    tick_fontsize = max(max_ticsize, 14 - map_size // 10)
    annot_fontsize = max(max_annotsize, 12 - map_size // 10)

    # Plot
    plt.title(title)
    sns.heatmap(
        attn_map,
        annot=annot,
        cmap="viridis",
        fmt=".2f",
        cbar=True,
        annot_kws={"fontsize": annot_fontsize},
    )
    plt.xlabel("Key Vertices")
    plt.ylabel("Query Vertices")
    if x_labels is None:
        x_labels = [f"Vertex {i}" for i in range(attn_map.shape[0])]
    if y_labels is None:
        y_labels = [f"Vertex {i}" for i in range(attn_map.shape[1])]

    plt.xticks(
        ticks=np.arange(attn_map.shape[1]) + 0.5,
        labels=x_labels,
        rotation=45,
        fontsize=tick_fontsize,
    )
    plt.yticks(
        ticks=np.arange(attn_map.shape[0]) + 0.5,
        labels=y_labels,
        rotation=0,
        fontsize=tick_fontsize,
    )
    plt.tight_layout()
    execution_function(
        plot_target=attn_map,
        fig=fig,
        title=title,
        figsize=figsize,
        x_labels=x_labels,
        y_labels=y_labels,
        fig_filename=attention_filename,
        **execution_kwargs,
    )
    return attn_map


def visualize_interactive_plot(**kwargs: Any) -> None:
    """
    Execution function for interactive visualization using Plotly.
    """
    interactive_fig = px.imshow(
        kwargs["plot_target"],
        text_auto=".2f",
        color_continuous_scale="Viridis",
        title=kwargs["title"],
        labels={"x": "X", "y": "Y"},
        width=int(kwargs["figsize"] * 100),
        height=int(kwargs["figsize"] * 100),
    )
    if "x_labels" in kwargs:
        interactive_fig.update_xaxes(
            tickvals=list(range(len(kwargs["x_labels"]))), ticktext=kwargs["x_labels"]
        )
    if "y_labels" in kwargs:
        interactive_fig.update_yaxes(
            tickvals=list(range(len(kwargs["y_labels"]))), ticktext=kwargs["y_labels"]
        )
    interactive_fig.show()
