"""
Stacked AutoEncoder (StackedAE) implementation.
"""

from typing import cast

import torch
from torch import nn

from .ae import AutoEncoder


class StackedAutoEncoder(nn.Module):
    """
    Stacked AutoEncoder - A stack of individual AutoEncoders trained layer-wise.
    """

    def __init__(self, layer_sizes: list[int], output_type: str = "prediction") -> None:
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

    def forward(
        self,
        x: torch.Tensor,
        return_embedding: bool | None = None,
        return_sequence: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass.
        If return_embedding is True (or output_type="embedding"), returns the latent code.
        Otherwise, returns reconstruction.
        """
        # Handle sequence input similar to AutoEncoder base
        is_sequence = x.dim() == 3
        b, s, f = 0, 0, 0
        if is_sequence:
            b, s, f = x.shape
            current = x.view(b * s, f)
        else:
            current = x

        # Encode path
        for ae_module in self.aes:
            ae = cast(AutoEncoder, ae_module)
            current = ae.encode(current)

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
            ae_back = cast(AutoEncoder, self.aes[i])
            decoded = ae_back.decode(decoded)

        out_final = decoded
        if is_sequence:
            out_final = out_final.view(b, s, -1)
            if not return_sequence:
                return out_final[:, -1, :]

        return out_final
