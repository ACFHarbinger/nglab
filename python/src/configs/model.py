
from __future__ import annotations

from dataclasses import dataclass

from python.src.configs.base import BaseConfig


@dataclass
class ModelConfig(BaseConfig):
    name: str = "base"
    seq_len: int = 30
    pred_len: int = 1
    embedding_dim: int = 128
    hidden_dim: int = 128
    dropout: float = 0.1
    output_type: str = "embedding"


@dataclass
class LSTMConfig(ModelConfig):
    name: str = "LSTM"
    n_encode_layers: int = 2
    return_sequence: bool = False
    seq_len: int = 30
