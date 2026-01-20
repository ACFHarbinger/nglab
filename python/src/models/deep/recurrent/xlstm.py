"""
xLSTM Model wrapping xLSTMBlocks.
"""

from torch import nn

from python.src.models.deep.modules.xlstm_block import xLSTMBlock


class xLSTM(nn.Module):
    """
    xLSTM Model (stacked sLSTM/mLSTM blocks) for Time Series or Sequence tasks.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        n_layers,
        output_dim,
        dropout=0.0,
        output_type="prediction",
        cell_type="slstm",
        num_heads=4,
    ):
        """
        Initialize the xLSTM.

        Args:
            input_dim (int): Input feature dimension.
            hidden_dim (int): Hidden state dimension.
            n_layers (int): Number of xLSTM layers.
            output_dim (int): Output dimension.
            dropout (float): Dropout probability.
            output_type (str): 'prediction' or 'embedding'.
            cell_type (str or list): 'slstm', 'mlstm', or list of types.
            num_heads (int): Number of heads for mLSTM cells.
        """
        super().__init__()
        self.output_type = output_type
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Determine cell types per layer
        if isinstance(cell_type, str):
            cell_types = [cell_type] * n_layers
        else:
            msg = "cell_type list length must match n_layers"
            assert len(cell_type) == n_layers, msg
            cell_types = cell_type

        # We can implement stacking manually or use a loop
        self.layers = nn.ModuleList(
            [
                xLSTMBlock(
                    input_dim=input_dim if i == 0 else hidden_dim,
                    hidden_dim=hidden_dim,
                    batch_first=True,
                    dropout=dropout,
                    cell_type=cell_types[i],
                    num_heads=num_heads,
                )
                for i in range(n_layers)
            ]
        )

        self.norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, return_embedding=None, return_sequence=False):
        """
        Forward pass.

        Args:
            x (Tensor): Input sequence [batch, seq_len, input_dim].
            return_embedding (bool, optional): Override output type.
            return_sequence (bool, optional): If True, return full sequence.
        """
        # x is [Batch, Seq, Feat] (since batch_first=True in blocks)

        current_state_list = [None] * self.n_layers

        for i, layer in enumerate(self.layers):
            x, state = layer(x, current_state_list[i])
            # x is output sequence of this layer
            current_state_list[i] = state

        # x is now the output sequence of the last layer
        x = self.norm(x)

        # Determine state
        if return_sequence:
            state = x
        else:
            state = x[:, -1, :]

        should_return_embedding = (
            return_embedding
            if return_embedding is not None
            else (self.output_type == "embedding")
        )

        if should_return_embedding:
            return state

        return self.fc(state)
