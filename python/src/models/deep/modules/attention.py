"""
Attention mechanisms for Time Series Models.
"""

from math import sqrt
from typing import Any

import numpy as np
import torch
from torch import nn

from python.src.utils.functions.masking import ProbMask, TriangularCausalMask


# Adapted from the Time-Series-Library (https://github.com/thuml/Time-Series-Library/blob/main/layers/SelfAttention_Family.py)
class DSAttention(nn.Module):
    """
    De-stationary Attention mechanism.
    """

    def __init__(
        self,
        mask_flag: bool = True,
        attention_dropout: float = 0.1,
        output_attention: bool = False,
        scale: float | None = None,
    ) -> None:
        """
        Initialize.
        """
        super().__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(  # noqa: PLR0913
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask: Any = None,
        tau: torch.Tensor | None = None,
        delta: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Forward pass.
        """
        B, L, _H, E = queries.shape
        _, _S, _, _D = values.shape
        scale = self.scale or 1.0 / sqrt(E)

        tau_val: float | torch.Tensor = (
            1.0 if tau is None else tau.unsqueeze(1).unsqueeze(1)
        )  # B x 1 x 1 x 1
        delta_val: float | torch.Tensor = (
            0.0 if delta is None else delta.unsqueeze(1).unsqueeze(1)
        )  # B x 1 x 1 x S

        # De-stationary Attention
        scores = torch.einsum("blhe,bshe->bhls", queries, keys) * tau_val + delta_val

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class FullAttention(nn.Module):
    """
    Standard Full Attention.
    """

    def __init__(
        self,
        mask_flag: bool = True,
        factor: int = 5,
        scale: float | None = None,
        attention_dropout: float = 0.1,
        output_attention: bool = False,
    ) -> None:
        """
        Initialize.
        """
        super().__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(  # noqa: PLR0913
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask: Any = None,
        tau: torch.Tensor | None = None,
        delta: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Forward pass.
        """
        B, L, _H, E = queries.shape
        _, _S, _, _D = values.shape
        scale = self.scale or 1.0 / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class ProbAttention(nn.Module):
    """
    Informer-style Probabilistic Attention.
    """

    def __init__(
        self,
        mask_flag: bool = True,
        factor: int = 5,
        scale: float | None = None,
        attention_dropout: float = 0.1,
        output_attention: bool = False,
    ) -> None:
        """
        Initialize.
        """
        super().__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(  # noqa: N802
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        sample_k: int,
        n_top: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k))
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        Q_reduce = Q[
            torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], M_top, :
        ]
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))

        return Q_K, M_top

    def _get_initial_context(self, V: torch.Tensor, L_Q: int) -> torch.Tensor:
        B, H, L_V, _D = V.shape
        if not self.mask_flag:
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:
            assert L_Q == L_V
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(  # noqa: PLR0913
        self,
        context_in: torch.Tensor,
        V: torch.Tensor,  # noqa: N803
        scores: torch.Tensor,
        index: torch.Tensor,
        L_Q: int,  # noqa: N803
        attn_mask: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, H, L_V, _D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)

        context_in[
            torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :
        ] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[
                torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :
            ] = attn
            return context_in, attns
        else:
            return context_in, None

    def forward(  # noqa: PLR0913
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask: Any = None,
        tau: torch.Tensor | None = None,
        delta: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Forward pass.
        """
        _B, L_Q, _H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = int(self.factor * np.ceil(np.log(L_K)).item())
        u = int(self.factor * np.ceil(np.log(L_Q)).item())

        U_part = min(U_part, L_K)
        u = min(u, L_Q)

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        scale = self.scale or 1.0 / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale

        context = self._get_initial_context(values, L_Q)
        context, attn = self._update_context(
            context, values, scores_top, index, L_Q, attn_mask
        )

        return context.contiguous(), attn


class AttentionLayer(nn.Module):
    """
    Attention Layer wrapping inner attention mechanisms.
    """

    def __init__(
        self,
        attention: nn.Module,
        d_model: int,
        n_heads: int,
        d_keys: int | None = None,
        d_values: int | None = None,
    ) -> None:
        """
        Initialize.
        """
        super().__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(  # noqa: PLR0913
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask: Any = None,
        tau: torch.Tensor | None = None,
        delta: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Forward pass.
        """
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries, keys, values, attn_mask, tau=tau, delta=delta
        )
        out = out.view(B, L, -1)
        return self.out_projection(out), attn
