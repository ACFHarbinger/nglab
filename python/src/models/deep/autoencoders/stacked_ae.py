"""
Stacked AutoEncoder (StackedAE) implementation.
"""

from torch import nn
from .ae import AutoEncoder


class StackedAutoEncoder(nn.Module):
    """
    Stacked AutoEncoder - A stack of individual AutoEncoders trained layer-wise.
    """

    def __init__(self, layer_sizes, output_type="prediction"):
        """
        Args:
            layer_sizes (list[int]): Sizes of the layers [input, h1, h2, ..., latent].
            output_type (str): "prediction" (reconstruction) or "embedding".
        """
        super().__init__()
        self.layer_sizes = layer_sizes
        self.output_type = output_type

        # Create a stack of shallow AutoEncoders (no internal hidden layers between steps)
        # AE_i maps dim[i] -> dim[i+1] (latent) -> dim[i] (recon)
        self.aes = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            # Shallow AE: input=sizes[i], hidden=[], latent=sizes[i+1]
            ae = AutoEncoder(
                input_dim=layer_sizes[i],
                hidden_dims=[],
                latent_dim=layer_sizes[i + 1],
                output_type="embedding",  # Internal AEs act as encoders in the stack
            )
            self.aes.append(ae)

    def forward(self, x, return_embedding=None, return_sequence=False):
        """
        Forward pass.
        If return_embedding is True (or output_type="embedding"), returns the latent code.
        Otherwise, returns reconstruction.
        """
        # Handle sequence input similar to AutoEncoder base
        is_sequence = x.dim() == 3
        if is_sequence:
            b, s, f = x.shape
            x_flat = x.view(b * s, f)
            current = x_flat
        else:
            current = x

        # Encode path
        encoded_features = []  # Store intermediate activations if needed
        for ae in self.aes:
            current = ae.encode(current)
            encoded_features.append(current)

        latent = current

        should_return_embedding = (
            return_embedding
            if return_embedding is not None
            else (self.output_type == "embedding")
        )

        if should_return_embedding:
            out = latent
            if is_sequence and not return_sequence:
                return out.view(b, s, -1)[:, -1, :]
            if is_sequence:
                return out.view(b, s, -1)
            return out

        # Decode path (reverse order)
        decoded = latent
        for i in reversed(range(len(self.aes))):
            decoded = self.aes[i].decode(decoded)

        out = decoded
        if is_sequence:
            out = out.view(b, s, -1)
            if not return_sequence:
                return out[:, -1, :]

        return out
