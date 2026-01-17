"""Ensemble models facade."""

from .ensemble.bagging import BaggingModel
from .ensemble.stacking import StackingModel
from .ensemble.voting import VotingModel, WeightedAverageModel

__all__ = [
    "BaggingModel",
    "StackingModel",
    "VotingModel",
    "WeightedAverageModel",
]
