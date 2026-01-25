
from __future__ import annotations

from typing import Any

import torch.nn as nn

from python.src.models.recurrent.esn import EchoStateNetwork as ESN
from python.src.models.recurrent.lsm import LSM
from python.src.models.recurrent.rnn import LSTM, GRU
from python.src.models.recurrent.tsmamba import TSMamba
from python.src.models.recurrent.xlstm import xLSTM
from python.src.models.factories.base import NeuralComponentFactory


class RecurrentFactory(NeuralComponentFactory):
    """Factory for recurrent neural networks."""

    @staticmethod
    def get_component(name: str, **kwargs: Any) -> nn.Module:
        """Get recurrent model by name."""
        name = name.lower()
        if "lstm" in name and "xlstm" not in name:
            return LSTM(**kwargs)
        elif "gru" in name:
            return GRU(**kwargs)
        elif "xlstm" in name:
            return xLSTM(**kwargs)
        elif "mamba" in name:
            return TSMamba(**kwargs)
        elif "esn" in name:
            return ESN(**kwargs)
        elif "lsm" in name:
            return LSM(**kwargs)
        else:
            raise ValueError(
                f"Unknown recurrent model: {name}. "
                f"Available: lstm, gru, xlstm, mamba, esn, lsm"
            )
