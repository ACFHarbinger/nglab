import pytest
import torch

from python.src.models.modules.distance_graph_convolution import (
    DistanceAwareGraphConvolution,
)


@pytest.fixture
def graph_setup():
    batch_size, num_nodes, in_c, out_c = 2, 4, 3, 5
    h = torch.randn(batch_size, num_nodes, in_c)
    adj = torch.tensor(
        [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]], dtype=torch.float32
    )
    dist = torch.tensor(
        [
            [0.0, 1.0, 2.0, 1.0],
            [1.0, 0.0, 1.0, 2.0],
            [2.0, 1.0, 0.0, 1.0],
            [1.0, 2.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    return h, adj, dist, in_c, out_c


def test_distance_aware_graph_convolution_init():
    conv = DistanceAwareGraphConvolution(3, 5, distance_influence="inverse")
    assert conv.in_channels == 3
    assert conv.out_channels == 5
    assert hasattr(conv, "alpha")

    conv_exp = DistanceAwareGraphConvolution(3, 5, distance_influence="exponential")
    assert hasattr(conv_exp, "temp")

    conv_learn = DistanceAwareGraphConvolution(3, 5, distance_influence="learnable")
    assert hasattr(conv_learn, "distance_transform")


def test_get_distance_weights(graph_setup):
    _, _, dist, in_c, out_c = graph_setup

    # Inverse
    conv = DistanceAwareGraphConvolution(in_c, out_c, distance_influence="inverse")
    weights = conv.get_distance_weights(dist)
    assert weights.shape == dist.shape
    assert weights[0, 1] < weights[0, 0]  # weights[0,0] is approx 1/eps

    # Exponential
    conv = DistanceAwareGraphConvolution(in_c, out_c, distance_influence="exponential")
    weights = conv.get_distance_weights(dist)
    assert torch.all(weights <= 1.0)

    # Learnable
    conv = DistanceAwareGraphConvolution(in_c, out_c, distance_influence="learnable")
    weights = conv.get_distance_weights(dist)
    assert weights.shape == dist.shape


def test_forward_aggregations(graph_setup):
    h, adj, dist, in_c, out_c = graph_setup

    # Sum
    conv_sum = DistanceAwareGraphConvolution(in_c, out_c, aggregation="sum")
    out = conv_sum(h, adj, dist)
    assert out.shape == (2, 4, out_c)

    # Mean
    conv_mean = DistanceAwareGraphConvolution(in_c, out_c, aggregation="mean")
    out = conv_mean(h, adj, dist)
    assert out.shape == (2, 4, out_c)

    # Max
    conv_max = DistanceAwareGraphConvolution(in_c, out_c, aggregation="max")
    out = conv_max(h, adj, dist)
    assert out.shape == (2, 4, out_c)


def test_forward_no_dist(graph_setup):
    h, adj, _, in_c, out_c = graph_setup
    conv = DistanceAwareGraphConvolution(in_c, out_c)
    out = conv(h, adj)  # dist_matrix will be adj
    assert out.shape == (2, 4, out_c)


def test_single_graph_forward(graph_setup):
    h, adj, dist, in_c, out_c = graph_setup
    h_single = h[0]
    conv = DistanceAwareGraphConvolution(in_c, out_c, aggregation="mean")
    out = conv.single_graph_forward(h_single, adj, dist)
    assert out.shape == (4, out_c)

    # Max aggregation single graph
    conv_max = DistanceAwareGraphConvolution(in_c, out_c, aggregation="max")
    out_max = conv_max.single_graph_forward(h_single, adj, dist)
    assert out_max.shape == (4, out_c)
