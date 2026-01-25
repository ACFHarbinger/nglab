from __future__ import annotations

from dataclasses import dataclass

from python.src.configs.base import BaseConfig

__all__ = ["DataConfig", "PolymarketConfig"]


@dataclass
class DataConfig(BaseConfig):
    data_path: str = "data/polymarket/"
    target_column: str = "price"
    batch_size: int = 32
    seq_len: int = 30
    pred_len: int = 1
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    normalize: str = "minmax"
    num_workers: int = 4
    format: str = "csv"
    streaming: bool = False
    add_technical_indicators: bool = False


@dataclass
class PolymarketConfig(DataConfig):
    pass
