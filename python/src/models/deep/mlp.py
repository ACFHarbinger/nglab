"""
Multi-Layer Perceptron (MLP) implementation.
"""

from torch import nn


class MLP(nn.Module):
    """
    Multi-layer Perceptron (FF / DFF).
    Supports a configurable number of hidden layers and activation functions.
    """

    def __init__(
        self,
        input_dim,
        hidden_dims,
        output_dim,
        dropout=0.0,
        activation="relu",
        output_type="prediction",
    ):
        """
        Args:
            input_dim (int): Dimension of input features.
            hidden_dims (list): List of hidden layer dimensions.
            output_dim (int): Dimension of output.
            dropout (float): Dropout probability.
            activation (str): Activation function name ('relu', 'tanh', 'gelu', 'sigmoid').
            output_type (str): 'prediction' or 'embedding'.
        """
        super().__init__()
        self.output_type = output_type

        layers = []
        last_dim = input_dim

        # Select activation
        act_fn = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "gelu": nn.GELU,
            "sigmoid": nn.Sigmoid,
        }.get(activation.lower(), nn.ReLU)

        for h_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, h_dim))
            layers.append(act_fn())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            last_dim = h_dim

        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(last_dim, output_dim)

    def forward(self, x, return_embedding=None, return_sequence=False):
        """
        x: (Batch, Features) or (Batch, Seq, Features)
        """
        if x.dim() == 3:
            # Apply per time step or flatten?
            # Existing backbone pattern usually handles per-step in RNNs,
            # but MLPs usually flatten or apply to last step.
            # Let's apply to all steps if 3D.
            b, s, f = x.shape
            x = x.view(b * s, f)
            emb = self.backbone(x)
            emb = emb.view(b, s, -1)
        else:
            emb = self.backbone(x)

        should_return_embedding = (
            return_embedding
            if return_embedding is not None
            else (self.output_type == "embedding")
        )

        if should_return_embedding:
            if not return_sequence and emb.dim() == 3:
                return emb[:, -1, :]
            return emb

        out = self.head(emb)
        if not return_sequence and out.dim() == 3:
            return out[:, -1, :]
        return out
