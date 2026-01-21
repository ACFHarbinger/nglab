import pytest
import numpy as np
import torch
import sys
from unittest.mock import MagicMock, patch, ANY

# Pre-mock plotting libraries in sys.modules to avoid ImportErrors and side effects
sys.modules["matplotlib"] = MagicMock()
sys.modules["matplotlib.pyplot"] = MagicMock()
sys.modules["matplotlib.axes"] = MagicMock()
sys.modules["matplotlib.collections"] = MagicMock()
sys.modules["matplotlib.patches"] = MagicMock()
sys.modules["seaborn"] = MagicMock()
sys.modules["networkx"] = MagicMock()
sys.modules["plotly"] = MagicMock()
sys.modules["plotly.express"] = MagicMock()

from python.src.utils.plot_utils import (
    draw_graph,
    plot_linechart,
    plot_tsp,
    plot_vehicle_routes,
    plot_attention_maps_wrapper,
    discrete_cmap
)

@pytest.fixture
def mock_plt():
    with patch("python.src.utils.plot_utils.plt") as mock:
        yield mock

@pytest.fixture
def mock_sns():
    with patch("python.src.utils.plot_utils.sns") as mock:
        yield mock

@pytest.fixture
def mock_px():
    with patch("python.src.utils.plot_utils.px") as mock:
        yield mock

@pytest.fixture
def mock_nx():
    with patch("python.src.utils.plot_utils.nx") as mock:
        yield mock

class TestPlotUtils:
    def test_draw_graph(self, mock_nx, mock_plt):
        dist_matrix = np.array([[0, 1], [1, 0]])
        mock_graph = MagicMock()
        mock_nx.from_numpy_array.return_value = mock_graph
        mock_graph.edges.return_value = []
        
        draw_graph(dist_matrix)
        
        mock_nx.from_numpy_array.assert_called_with(dist_matrix)
        mock_nx.spring_layout.assert_called()
        mock_nx.draw.assert_called()
        mock_plt.show.assert_called()

    def test_plot_linechart_simple(self, mock_plt):
        # shape needs to support index 5 access (at least 6 columns)
        # Assuming: col 0 is x, col 5 is y
        row1 = [0, 1, 2, 3, 4, 5]
        row2 = [1, 2, 3, 4, 5, 6]
        graph_log = np.array([row1, row2])
        plot_func = MagicMock()
        
        plot_linechart(
            output_dest="test.png",
            graph_log=graph_log,
            plot_func=plot_func,
            policies=["p1"],
            fsave=False
        )
        
        mock_plt.figure.assert_called()
        plot_func.assert_called()
        mock_plt.show.assert_called()

    def test_plot_tsp(self, mock_plt):
        xy = np.array([[0, 0], [1, 1], [0, 1]])
        tour = np.array([0, 1, 2])
        ax = MagicMock()
        
        plot_tsp(xy, tour, ax)
        
        ax.scatter.assert_called()
        ax.plot.assert_called()
        ax.quiver.assert_called()

    def test_discrete_cmap(self, mock_plt):
        mock_plt.cm.get_cmap.return_value = MagicMock()
        cmap = discrete_cmap(5)
        assert cmap is not None

    def test_plot_vehicle_routes(self, mock_plt):
        data = {
            "depot": torch.tensor([0.5, 0.5]),
            "loc": torch.tensor([[0.1, 0.1], [0.9, 0.9]]),
            "demand": torch.tensor([1, 1])
        }
        route = torch.tensor([1, 2, 0])
        ax = MagicMock()
        
        plot_vehicle_routes(data, route, ax)
        
        ax.plot.assert_called()
        ax.quiver.assert_called()

    def test_plot_attention_maps_wrapper(self, mock_plt, mock_sns, tmp_path):
        model_name = "test_model"
        # Mock attention dict structure: [layer][head][batch] -> tensor
        # But code expects: attention_dict[model_name][sample_idx]["attention_weights"] -> tensor [layers, heads, batches]?
        # Re-reading code:
        # attention_weights = attention_dict[model_name][sample_idx]["attention_weights"]
        # shape expected: [layers, heads, batches] if extracting map.
        # then attn_map = attention_weights[layer_idx, head_idx, batch_idx]...
        
        # Let's create a mocked tensor
        attn_tensor = MagicMock()
        # Mock .cpu().numpy() chain
        attn_tensor.cpu.return_value.numpy.return_value = np.zeros((10, 10))
        # Support indexing
        attn_tensor.__getitem__.return_value = attn_tensor # simplify
        # Support shape
        attn_tensor.shape = (4, 8, 2) # layers, heads, batches
        
        attention_dict = {
            model_name: [
                {"attention_weights": attn_tensor}
            ]
        }
        
        exec_func = MagicMock()
        
        plot_attention_maps_wrapper(
            dir_path=str(tmp_path),
            attention_dict=attention_dict,
            model_name=model_name,
            execution_function=exec_func,
            layer_idx=0,
            head_idx=0,
            batch_idx=0
        )
        
        mock_sns.heatmap.assert_called()
        exec_func.assert_called()
