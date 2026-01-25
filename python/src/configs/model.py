from dataclasses import dataclass

from omegaconf import MISSING


@dataclass
class ModelConfig:
    name: str = MISSING
    # Use interpolation syntax for defaults that depend on other config groups
    # However, in dataclass definition, we often set specific defaults
    # or rely on Hydra to fill them if they are MISSING
    seq_len: int = 30  # Default, can be overridden by interpolation
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
    # To support ${env.lookback}, we can't easily put it in the dataclass default
    # unless we use a string and cast it later, or rely on the composition.
    # For now we use the static default of 30 matching env default.
    seq_len: int = 30
