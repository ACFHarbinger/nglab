"""Optimized Graph Convolution implementation with multiple aggregators."""

from collections.abc import Iterable
from typing import Any, cast

import torch
from torch import Tensor
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import add_remaining_self_loops, scatter

# Handle optional SparseTensor
try:
    from torch_geometric.typing import Adj, OptTensor, SparseTensor, torch_sparse
except ImportError:
    Adj = Any
    OptTensor = Any
    SparseTensor = Any
    torch_sparse = Any


# Adapted from https://github.com/shyam196/egc
class EfficientGraphConvolution(MessagePassing):
    """
    Efficient Graph Convolution (EGC) with multiple aggregators.

    This layer computes node updates using a linear combination of different
    neighborhood aggregations (mean, max, sum, var, std, symnorm) and self-features.
    Supports multi-head weights and basis functions for efficiency.
    """

    _cached_edge_index: tuple[Any, Any] | None
    _cached_adj_t: Any | None

    def __init__(  # noqa: PLR0913
        self,
        in_channels: int,
        out_channels: int,
        aggrs: Iterable[str] = ("symnorm",),
        num_heads: int = 8,
        num_bases: int = 4,
        cached: bool = False,
        add_self_loops: bool = True,
        bias: bool = True,
        sigmoid: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            in_channels: Dimension of input features.
            out_channels: Dimension of output features.
            aggrs: Iterable of aggregator names to use (e.g., "sum", "mean", "symnorm").
            num_heads: Number of attention heads.
            num_bases: Number of basis functions for the weight matrix.
            cached: If set to `True`, the layer will cache the computation of
                :obj:`edge_index` and :obj:`symnorm_weight` on first execution,
                and will use the cached values for further executions.
            add_self_loops: If set to `False`, will not add self-loops to the
                input graph.
            bias: Whether to use a bias term.
            sigmoid: If set to `True`, applies a sigmoid activation to the weighting coefficients.
        """
        super().__init__(node_dim=1, **kwargs)
        if out_channels % num_heads != 0:
            raise ValueError("out_channels must be divisible by the number of heads")

        for a in aggrs:
            if a not in {"sum", "mean", "symnorm", "min", "max", "var", "std"}:
                raise ValueError(f"Unsupported aggregator: {a}")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.num_bases = num_bases
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.aggregators = list(aggrs)
        self.sigmoid = sigmoid

        self.bases_weight = Parameter(
            torch.Tensor(in_channels, (out_channels // num_heads) * num_bases)
        )
        self.comb_weight = Linear(
            in_channels, num_heads * num_bases * len(self.aggregators)
        )

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Resets the parameters of the layer using Glorot initialization."""
        glorot(self.bases_weight)
        self.comb_weight.reset_parameters()
        zeros(self.bias)
        self._cached_adj_t = None
        self._cached_edge_index = None

    def forward(self, x: Tensor, edge_index: Any) -> Tensor:  # noqa: PLR0915
        """
        Forward pass for Efficient Graph Convolution.

        Args:
            x: Node features tensor of shape (batch_size, num_nodes, in_channels).
            edge_index: Graph adjacency information.

        Returns:
            Updated node features tensor.
        """
        symnorm_weight: OptTensor = None
        if "symnorm" in self.aggregators:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, symnorm_weight = gcn_norm(
                        edge_index,
                        None,
                        num_nodes=x.size(self.node_dim),
                        improved=False,
                        add_self_loops=self.add_self_loops,
                    )
                    if self.cached:
                        self._cached_edge_index = (
                            cast(Tensor, edge_index),
                            symnorm_weight,
                        )
                else:
                    edge_index, symnorm_weight = cache

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(
                        edge_index,
                        None,
                        num_nodes=x.size(self.node_dim),
                        improved=False,
                        add_self_loops=self.add_self_loops,
                    )
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        elif self.add_self_loops:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if self.cached and cache is not None:
                    edge_index = cache[0]
                else:
                    edge_index, _ = add_remaining_self_loops(edge_index)
                    if self.cached:
                        self._cached_edge_index = (edge_index, None)

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if self.cached and cache is not None:
                    edge_index = cache
                else:
                    edge_index = torch_sparse.fill_diag(edge_index, 1.0)
                    if self.cached:
                        self._cached_adj_t = edge_index

        batch_size = x.size(0)
        num_nodes = x.size(1)

        # [num_nodes, (out_channels // num_heads) * num_bases]
        bases = torch.matmul(x, self.bases_weight)
        # [num_nodes, num_heads * num_bases * num_aggrs]
        weightings = self.comb_weight(x)
        if self.sigmoid:
            weightings = torch.sigmoid_(weightings)

        if symnorm_weight is not None:
            symnorm_weight = symnorm_weight.view(-1, 1)

        # [num_nodes, num_aggregators, (out_channels // num_heads) * num_bases]
        # propagate_type: (x: Tensor, symnorm_weight: OptTensor)
        aggregated = self.propagate(
            edge_index, x=bases, symnorm_weight=symnorm_weight, size=None
        )

        weightings = weightings.view(
            batch_size,
            num_nodes,
            self.num_heads,
            self.num_bases * len(self.aggregators),
        )
        aggregated = aggregated.view(
            batch_size,
            num_nodes,
            len(self.aggregators) * self.num_bases,
            self.out_channels // self.num_heads,
        )

        # [num_nodes, num_heads, out_channels // num_heads]
        out = torch.matmul(weightings, aggregated)
        out = out.view(batch_size, num_nodes, self.out_channels)
        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_j: Tensor) -> Tensor:
        """
        Passes messages along edges.
        """
        return x_j

    def aggregate(
        self,
        inputs: Tensor,
        index: Tensor,
        ptr: OptTensor = None,
        dim_size: int | None = None,
        **kwargs: Any,
    ) -> Tensor:
        """
        Aggregates messages from neighbors using multiple aggregators.
        """
        symnorm_weight = kwargs.get("symnorm_weight")
        aggregated = []
        inputs = inputs.permute(1, 0, 2)
        for aggregator in self.aggregators:
            if aggregator == "sum":
                out = scatter(inputs, index, 0, dim_size, reduce="sum")
            elif aggregator == "symnorm":
                assert symnorm_weight is not None
                out = scatter(
                    inputs * symnorm_weight.unsqueeze(-1),
                    index,
                    0,
                    dim_size,
                    reduce="sum",
                )
            elif aggregator == "mean":
                out = scatter(inputs, index, 0, dim_size, reduce="mean")
            elif aggregator == "min":
                out = scatter(inputs, index, 0, dim_size, reduce="min")
            elif aggregator == "max":
                out = scatter(inputs, index, 0, dim_size, reduce="max")
            elif aggregator in {"var", "std"}:
                mean = scatter(inputs, index, 0, dim_size, reduce="mean")
                mean_squares = scatter(
                    inputs * inputs, index, 0, dim_size, reduce="mean"
                )
                out = mean_squares - mean * mean
                if aggregator == "std":
                    out = torch.sqrt(torch.relu(out) + 1e-5)
            else:
                raise ValueError(f'Unknown aggregator "{aggregator}".')
            aggregated.append(out)

        return torch.stack(aggregated, dim=1)

    def message_and_aggregate(self, edge_index: Any, **kwargs: Any) -> Tensor:
        """
        Performs message passing and aggregation in a single step for sparse tensors.
        """
        x = kwargs.get("x")
        if x is None:
            raise ValueError("x must be passed to message_and_aggregate")
        aggregated = []
        if len(self.aggregators) > 1 and "symnorm" in self.aggregators:
            adj_t_nonorm = edge_index.set_value(None)
        else:
            # No normalization is calculated in forward if symnorm isn't one
            # of the aggregators
            adj_t_nonorm = edge_index

        for aggregator in self.aggregators:
            if aggregator == "symnorm":
                correct_adj = edge_index
                agg = "sum"
            else:
                correct_adj = adj_t_nonorm
                agg = aggregator

            if aggregator in ["var", "std"]:
                mean = torch_sparse.matmul(correct_adj, x, reduce="mean")
                mean_sq = torch_sparse.matmul(correct_adj, x * x, reduce="mean")
                out = mean_sq - mean * mean
                if aggregator == "std":
                    out = torch.sqrt(torch.relu(out) + 1e-5)
                aggregated.append(out)
            else:
                aggregated.append(torch_sparse.matmul(correct_adj, x, reduce=agg))

        return torch.stack(aggregated, dim=1)

    def __repr__(self) -> str:
        """String representation of the layer."""
        return f"{self.__class__.__name__}({self.in_channels}, {self.out_channels}, {self.aggregators})"
