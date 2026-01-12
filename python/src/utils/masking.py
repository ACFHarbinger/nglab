"""
Masking utilities for self-attention.
"""
import torch


#  Adapted from the Time-Series-Library (https://github.com/thuml/Time-Series-Library/blob/main/utils/masking.py)
class TriangularCausalMask():
    """
    Triangular causal mask for sequence processing.
    """
    def __init__(self, B, L, device="cpu"):
        """
        Initialize the mask.

        Args:
            B (int): Batch size.
            L (int): Sequence length.
            device (str): Device to use.
        """
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        """Return the mask tensor."""
        return self._mask


class ProbMask():
    """
    Probabilistic mask for Informer-style attention.
    """
    def __init__(self, B, H, L, index, scores, device="cpu"):
        """
        Initialize the mask.
        """
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        """
        Return the mask tensor.
        """
        return self._mask