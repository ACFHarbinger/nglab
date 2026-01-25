from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

# Do NOT mock sys.modules here as it breaks other tests.
# Instead, patch inside the tests or fixtures.
from python.src.utils.plot_utils import (
    discrete_cmap,
    draw_graph,
    plot_attention_maps_wrapper,
    plot_linechart,
    visualize_interactive_plot,
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
        mock.from_numpy_array.return_value = MagicMock()
        yield mock


class TestPlotUtils:
    def test_draw_graph(self, mock_nx, mock_plt):
        dist_matrix = np.array([[0, 1], [1, 0]])
        mock_graph = MagicMock()
        mock_nx.from_numpy_array.return_value = mock_graph
        mock_graph.edges.return_value = [(0, 1, {"weight": 1})]

        draw_graph(dist_matrix)

        mock_nx.from_numpy_array.assert_called_with(dist_matrix)
        mock_nx.spring_layout.assert_called()
        mock_nx.draw.assert_called()
        mock_plt.show.assert_called()

    def test_plot_linechart_simple(self, mock_plt):
        row1 = [0, 1, 2, 3, 4, 5]
        row2 = [1, 2, 3, 4, 5, 6]
        graph_log = np.array([row1, row2])
        plot_func = MagicMock()

        plot_linechart(
            output_dest="test.png",
            graph_log=graph_log,
            plot_func=plot_func,
            policies=["p1"],
            fsave=False,
        )

        mock_plt.figure.assert_called()
        plot_func.assert_called()
        mock_plt.show.assert_called()

    def test_plot_linechart_pareto(self, mock_plt):
        # graph_log shape [n_policies, dim1, dim2]
        # e.g. [1, 2, 6] -> 1 policy, 2 points, each point has 6 values (0=x, 5=y)
        graph_log = np.zeros((1, 2, 6))
        graph_log[0, 0, 0] = 1.0
        graph_log[0, 0, 5] = 10.0  # Point 1
        graph_log[0, 1, 0] = 2.0
        graph_log[0, 1, 5] = 5.0  # Point 2

        plot_func = MagicMock()

        dominants = plot_linechart(
            output_dest="pareto.png",
            graph_log=graph_log,
            plot_func=plot_func,
            policies=["p1"],
            pareto_front=True,
            fsave=False,
        )

        assert dominants is not None
        assert len(dominants) == 2  # 2 policies
        assert dominants[0] == [1]
        # Actually (1,10) vs (2,5). (1,10) is better on y, (2,5) is better on x?
        # Re-check logic: other[0] <= point[0] and other[1] >= point[1]
        # (1, 10) vs (2, 5): 1 <= 2 but 10 not <= 5. Not dominated.

        mock_plt.plot.assert_called()  # Should plot pareto front

    def test_discrete_cmap(self, mock_plt):
        mock_plt.cm.get_cmap.return_value = MagicMock()
        cmap = discrete_cmap(5)
        assert cmap is not None

    def test_plot_attention_maps_wrapper(self, mock_plt, mock_sns, tmp_path):
        model_name = "test_model"
        attn_tensor = MagicMock()
        attn_tensor.cpu.return_value.numpy.return_value = np.zeros((10, 10))
        attn_tensor.__getitem__.return_value = attn_tensor
        attn_tensor.shape = (4, 8, 2)

        attention_dict = {model_name: [{"attention_weights": attn_tensor}]}

        exec_func = MagicMock()

        plot_attention_maps_wrapper(
            dir_path=str(tmp_path),
            attention_dict=attention_dict,
            model_name=model_name,
            execution_function=exec_func,
            layer_idx=0,
            head_idx=0,
            batch_idx=0,
        )

        mock_sns.heatmap.assert_called()
        exec_func.assert_called()

    def test_plot_attention_maps_avg(self, mock_plt, mock_sns, tmp_path):
        model_name = "test_model"
        attn_tensor = torch.randn(4, 8, 2, 10, 10)  # L, H, B, V, V

        attention_dict = {model_name: [{"attention_weights": attn_tensor}]}
        exec_func = MagicMock()

        # Average heads and batches
        plot_attention_maps_wrapper(
            dir_path=str(tmp_path),
            attention_dict=attention_dict,
            model_name=model_name,
            execution_function=exec_func,
            head_idx=-1,
            batch_idx=-1,
        )

        mock_sns.heatmap.assert_called()

    @patch("python.src.utils.plot_utils.px")
    def test_visualize_interactive_plot(self, mock_px):
        kwargs = {
            "plot_target": np.zeros((10, 10)),
            "title": "Interactive",
            "figsize": 6.0,
            "x_labels": ["a"],
            "y_labels": ["b"],
        }
        visualize_interactive_plot(**kwargs)
        mock_px.imshow.assert_called()
        mock_px.imshow.return_value.show.assert_called()
