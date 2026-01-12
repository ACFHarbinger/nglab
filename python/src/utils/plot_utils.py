"""
Plotting utilities for the WSmart+ Route framework.

This module provides functions for generating various plots:
- Static line charts (training curves, Pareto fronts).
- TSP and VRP route visualizations using Matplotlib.
- Attention map heatmaps.
- Interactive visualizations.
"""

import os
import math
import numpy as np
import networkx as nx
import seaborn as sns
import plotly.express as px

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from logic.src.utils.io_utils import compose_dirpath





def draw_graph(distance_matrix):
    """
    Draws a networkx graph from a distance matrix using spring layout.

    Args:
        distance_matrix (np.ndarray): The adjacency/distance matrix.
    """
    G = nx.from_numpy_array(distance_matrix)
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True)
    labels = {(u, v): str(attr["weight"]) for u, v, attr in G.edges(data=True)}
    print(G.edges(data=True))
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.show()
    return


def plot_linechart(
    output_dest,
    graph_log,
    plot_func,
    policies,
    x_label=None,
    y_label=None,
    title=None,
    fsave=True,
    scale="linear",
    x_values=None,
    linestyles=None,
    markers=None,
    annotate=True,
    pareto_front=False,
):
    """
    Plots a generic line chart, optionally handling multiple policies and Pareto fronts.

    Args:
        output_dest (str): File path to save the plot.
        graph_log (np.ndarray): Data to plot.
        plot_func (callable): The pyplot plotting function (e.g., plt.plot, plt.scatter).
        policies (list): List of policy names for the legend.
        x_label (str, optional): Label for x-axis.
        y_label (str, optional): Label for y-axis.
        title (str, optional): Chart title.
        fsave (bool, optional): Whether to save figure to file. Defaults to True.
        scale (str, optional): Axis scale ('linear', 'log'). Defaults to "linear".
        x_values (list, optional): Custom x-values.
        linestyles (list, optional): List of linestyles.
        markers (list, optional): List of markers.
        annotate (bool, optional): Whether to annotate points. Defaults to True.
        pareto_front (bool, optional): Whether to calculate and overlay Pareto front. Defaults to False.

    Returns:
        list or None: Pareto dominants if pareto_front is True, else None.
    """

    def plot_graphs_out(plot_func, graph_log, x_values, linestyles, markers):
        """
        Helper to plot graphs for different policies.

        Args:
            plot_func: Function to plot a single line.
            graph_log: Log data.
            x_values: X-axis values.
            linestyles: Line styles.
            markers: Markers.
        """
        points_by_nbins = {}
        for id, lg in enumerate(zip(*graph_log)):
            if x_values is None:
                to_plot = (*lg,)
            else:
                to_plot = (
                    x_values,
                    *lg,
                )

            if linestyles is not None:
                line = linestyles[id % len(linestyles)]
            else:
                line = False

            if markers is not None:
                mark = markers[id % len(markers)]
            else:
                mark = False

            if not line and not mark:
                plot_func(*to_plot)
            elif not mark:
                plot_func(*to_plot, linestyle=line)
            elif not line:
                plot_func(*to_plot, marker=mark)
            else:
                plot_func(*to_plot, linestyle=line, marker=mark)

            for id, (x, y) in enumerate(zip(list(zip(*lg))[0], list(zip(*lg))[5])):
                if id not in points_by_nbins:
                    points_by_nbins[id] = []
                points_by_nbins[id].append((x, y))
        return points_by_nbins

    plt.figure(dpi=200)
    if title is not None:
        plt.title(title)

    if len(graph_log.shape) == 2:
        points_by_nbins = {0: []}
        for id, ll in enumerate(graph_log):
            if markers is not None:
                mark = markers[id % len(markers)]
                plot_func(ll, mark)
            else:
                plot_func(ll)

            points_by_nbins[0].append((ll[0], ll[5]))
    elif len(graph_log.shape) == 3:
        # nbins = [20, 50, 100, 200]
        # for id in range(len(nbins)):
        #    graph_log[id, :, 0] /= nbins[id]
        points_by_nbins = plot_graphs_out(
            plot_func, graph_log, x_values, linestyles, markers
        )
        if annotate:
            for lg in zip(*graph_log):
                for id, xy in enumerate(zip(list(zip(*lg))[0], list(zip(*lg))[5])):
                    # plt.annotate(id, xy=xy, textcoords='data')
                    if id == graph_log.shape[0] - 1:
                        plt.scatter(
                            xy[0],
                            xy[1],
                            s=200,
                            marker="o",
                            facecolors="none",
                            edgecolors="black",
                            linewidths=1,
                            zorder=10,
                        )

    # Pareto front for minimizing x and maximizing y
    if pareto_front:
        pareto_dominants = []
        pareto_labels = []
        for id_nbins, points in points_by_nbins.items():
            pareto_points = []
            dominance_ls = [0] * len(points)
            for id_point, point in enumerate(points):
                dominated = False
                for other_point in points:
                    # Check if other_point dominates point
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

            # Sort Pareto points by x-coordinate for drawing the front
            pareto_points.sort(key=lambda p: p[0])

            pareto_x = [p[0] for p in pareto_points]
            pareto_y = [p[1] for p in pareto_points]

            # colors = ['red', 'blue', 'green', 'purple', 'orange']
            # color = colors[id % len(colors)]

            label = f"Pareto Front (ID {id_nbins})"
            pareto_labels.append(label)
            plt.plot(
                pareto_x,
                pareto_y,
                "--",
                color="black",
                linewidth=2,
                label=label,
                zorder=5,
            )

            # Add dots at Pareto points
            # plt.scatter(pareto_x, pareto_y, c=color, s=50, zorder=6)

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


# Code inspired by Google OR Tools plot:
# https://github.com/google/or-tools/blob/fb12c5ded7423d524fc6c95656a9bdc290a81d4d/examples/python/cvrptw_plot.py
def plot_tsp(xy, tour, ax1):
    """
    Plot the TSP tour on matplotlib axis ax1.

    Args:
        xy (np.ndarray): Node coordinates [N, 2].
        tour (np.ndarray): Tour indices [N+1] (including return to start).
        ax1 (matplotlib.axes.Axes): The axis to plot on.
    """
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    xs, ys = xy[tour].transpose()
    xs, ys = xy[tour].transpose()
    dx = np.roll(xs, -1) - xs
    dy = np.roll(ys, -1) - ys
    d = np.sqrt(dx * dx + dy * dy)
    lengths = d.cumsum()

    # Scatter nodes
    ax1.scatter(xs, ys, s=40, color="blue")
    # Starting node
    ax1.plot(xy[0][0], xy[0][1], "sk", markersize=15, zorder=5)
    ax1.scatter([xs[0]], [ys[0]], s=100, color="red", zorder=4)

    # Arcs
    ax1.quiver(
        xs,
        ys,
        dx,
        dy,
        scale_units="xy",
        angles="xy",
        scale=1,
    )

    ax1.set_title("{} nodes, total length {:.2f}".format(len(tour), lengths[-1]))


def discrete_cmap(N, base_cmap=None):
    """
    Create an N-bin discrete colormap from the specified input map.

    Args:
        N (int): Number of bins.
        base_cmap (str or Colormap, optional): Base colormap name.

    Returns:
        Colormap: Discretized colormap.
    """
    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:
    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)


def plot_vehicle_routes(
    data,
    route,
    ax1,
    markersize=5,
    visualize_demands=False,
    demand_scale=1,
    round_demand=False,
):
    """
    Plot the vehicle routes on matplotlib axis ax1.

    Args:
        data (dict): Dictionary with 'depot', 'loc', 'demand'.
        route (Tensor): Route indices (single sequence with delimiters).
        ax1 (matplotlib.axes.Axes): Axis to plot on.
        markersize (int, optional): Size of markers. Defaults to 5.
        visualize_demands (bool, optional): Visualize demands as bars. Defaults to False.
        demand_scale (float, optional): Scaling factor for demands. Defaults to 1.
        round_demand (bool, optional): Round demand values in labels. Defaults to False.
    """
    # Route is one sequence, separating different routes with 0 (depot)
    routes = [
        r[r != 0]
        for r in np.split(route.cpu().numpy(), np.where(route == 0)[0])
        if (r != 0).any()
    ]
    depot = data["depot"].cpu().numpy()
    locs = data["loc"].cpu().numpy()
    demands = data["demand"].cpu().numpy() * demand_scale
    capacity = demand_scale  # capacity is always 1

    x_dep, y_dep = depot
    ax1.plot(x_dep, y_dep, "sk", markersize=markersize * 4)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.legend(loc="upper center")

    cmap = discrete_cmap(len(routes) + 2, "nipy_spectral")
    dem_rects = []
    used_rects = []
    cap_rects = []
    qvs = []
    total_dist = 0
    for veh_number, r in enumerate(routes):
        color = cmap(len(routes) - veh_number)  # invert to have in rainbow order

        route_demands = demands[r - 1]
        coords = locs[r - 1, :]
        xs, ys = coords.transpose()

        total_route_demand = sum(route_demands)
        assert total_route_demand <= capacity
        if not visualize_demands:
            ax1.plot(xs, ys, "o", mfc=color, markersize=markersize, markeredgewidth=0.0)

        dist = 0
        x_prev, y_prev = x_dep, y_dep
        cum_demand = 0
        for (x, y), d in zip(coords, route_demands):
            dist += np.sqrt((x - x_prev) ** 2 + (y - y_prev) ** 2)

            cap_rects.append(Rectangle((x, y), 0.01, 0.1))
            used_rects.append(
                Rectangle((x, y), 0.01, 0.1 * total_route_demand / capacity)
            )
            dem_rects.append(
                Rectangle(
                    (x, y + 0.1 * cum_demand / capacity), 0.01, 0.1 * d / capacity
                )
            )

            x_prev, y_prev = x, y
            cum_demand += d

        dist += np.sqrt((x_dep - x_prev) ** 2 + (y_dep - y_prev) ** 2)
        total_dist += dist
        qv = ax1.quiver(
            xs[:-1],
            ys[:-1],
            xs[1:] - xs[:-1],
            ys[1:] - ys[:-1],
            scale_units="xy",
            angles="xy",
            scale=1,
            color=color,
            label="R{}, # {}, c {} / {}, d {:.2f}".format(
                veh_number,
                len(r),
                int(total_route_demand) if round_demand else total_route_demand,
                int(capacity) if round_demand else capacity,
                dist,
            ),
        )

        qvs.append(qv)

    ax1.set_title("{} routes, total distance {:.2f}".format(len(routes), total_dist))
    ax1.legend(handles=qvs)

    pc_cap = PatchCollection(
        cap_rects, facecolor="whitesmoke", alpha=1.0, edgecolor="lightgray"
    )
    pc_used = PatchCollection(
        used_rects, facecolor="lightgray", alpha=1.0, edgecolor="lightgray"
    )
    pc_dem = PatchCollection(dem_rects, facecolor="black", alpha=1.0, edgecolor="black")

    if visualize_demands:
        ax1.add_collection(pc_cap)
        ax1.add_collection(pc_used)
        ax1.add_collection(pc_dem)


@compose_dirpath
def plot_attention_maps_wrapper(
    dir_path,
    attention_dict,
    model_name,
    execution_function,
    layer_idx=0,
    sample_idx=0,
    head_idx=0,
    batch_idx=0,
    x_labels=None,
    y_labels=None,
    **execution_kwargs,
):
    """
    Plot attention maps as heatmaps for a given layer, head, batch, and simulation sample.

    Args:
        dir_path (str): Directory path to save the heatmap image.
        attention_dict (dict): Dictionary where:
                              - Keys are model names (str);
                              - Values are lists of attention data for each sample, where each element is a dictionary containing:
                                'attention_weights' tensor of shape [num_layers, n_heads, batch_size, graph_size, graph_size].
        model_name (str): Name of the model to extract attention maps for.
        execution_function (function): Function that handles the plotting/saving logic.
        layer_idx (int): Index of the layer to visualize.
        sample_idx (int): Index of the simulation sample to visualize.
        head_idx (int): Index of the head to visualize (-1 for average over all heads).
        batch_idx (int): Index of the data batch to visualize (-1 for average over all batches).
        x_labels (list, optional): Custom labels for x-axis vertices.
        y_labels (list, optional): Custom labels for y-axis vertices.
        **execution_kwargs: Additional arguments to pass to the execution function.

    Returns:
        attn_map (np.ndarray): The attention map as a Numpy array.
    """
    assert sample_idx >= 0, f"sample_idx {sample_idx} must be a non-negative integer"

    attention_weights = attention_dict[model_name][sample_idx]["attention_weights"]
    assert (
        layer_idx < attention_weights.shape[0]
    ), f"layer_idx {layer_idx} exceeds number of layers {attention_weights.shape[0]}"
    assert (
        head_idx < attention_weights.shape[1]
    ), f"head_idx {head_idx} exceeds number of heads {attention_weights.shape[1]}"
    assert (
        batch_idx < attention_weights.shape[2]
    ), f"layer_idx {batch_idx} exceeds batch size {attention_weights.shape[2]}"

    # Extract attention map
    if head_idx >= 0:
        if batch_idx >= 0:
            attn_map = attention_weights[layer_idx, head_idx, batch_idx].cpu().numpy()
            title = "Attention Map (Layer {}, Head {}, Batch {})".format(
                layer_idx, head_idx, batch_idx
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
            title = "Attention Map Average Over All Batches (Layer {}, Head {})".format(
                layer_idx, head_idx
            )
            attention_filename = os.path.join(
                dir_path,
                "attention_maps",
                model_name,
                f"layer{layer_idx}_head{head_idx}_map{sample_idx}.png",
            )
    else:
        if batch_idx >= 0:
            attn_map = (
                attention_weights[layer_idx, :, batch_idx].mean(dim=0).cpu().numpy()
            )  # Average over heads
            title = "Attention Map Average Over All Heads (Layer {}, Batch {})".format(
                layer_idx, batch_idx
            )
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
            title = (
                "Attention Map Average Over All Heads and Batches (Layer {})".format(
                    layer_idx
                )
            )
            attention_filename = os.path.join(
                dir_path,
                "attention_maps",
                model_name,
                f"layer{layer_idx}_headavg_map{sample_idx}.png",
            )

    try:
        os.makedirs(os.path.dirname(attention_filename), exist_ok=True)
    except Exception:
        raise Exception(
            "directories to save attention maps do not exist and could not be created"
        )

    # Dynamically set figure size based on map_size
    base_vertexsize = 0.5
    map_size = math.isqrt(attn_map.shape[0] * attn_map.shape[1])
    min_figsize = 6.0
    max_figsize = 30.0
    figsize = min(max(min_figsize, base_vertexsize * map_size), max_figsize)
    fig = plt.figure(figsize=(figsize, figsize))

    # Adjust annotations and font sizes to scale inversely with map_size
    max_ticsize = 8
    max_annotsize = 8
    annot = (
        True if map_size <= 55 else False
    )  # Disable annotations for large graphs to avoid clutter
    tick_fontsize = max(max_ticsize, 14 - map_size // 10)
    annot_fontsize = max(max_annotsize, 12 - map_size // 10)

    # Plot and/or log attention heatmap
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
        ticks=range(attn_map.shape[0]),
        labels=x_labels,
        rotation=45,
        fontsize=tick_fontsize,
    )
    plt.yticks(
        ticks=range(attn_map.shape[1]),
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


def visualize_interactive_plot(**kwargs):
    """
    Execution function for interactive visualization using Plotly.

    Args:
        **kwargs: Keyword arguments containing 'plot_target', 'title', 'x_labels', 'y_labels', 'figsize'.
    """
    interactive_fig = px.imshow(
        kwargs["plot_target"],
        text_auto=".2f",
        color_continuous_scale="Viridis",
        title=kwargs["title"],
        labels={"x": kwargs["x_labels"], "y": kwargs["y_labels"]},
        width=kwargs["figsize"],
        height=kwargs["figsize"],
    )
    interactive_fig.update_xaxes(
        tickvals=list(range(len(kwargs["x_labels"]))), ticktext=kwargs["x_labels"]
    )
    interactive_fig.update_yaxes(
        tickvals=list(range(len(kwargs["y_labels"]))), ticktext=kwargs["y_labels"]
    )
    interactive_fig.show()
    return
