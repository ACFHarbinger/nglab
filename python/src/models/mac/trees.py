"""Tree-based models facade."""

from .trees.decision_tree import DecisionTreeModel
from .trees.random_forest import RandomForestModel
from .trees.cart import CARTModel
from .trees.id3 import ID3Model
from .trees.c45 import C45Model
from .trees.c50 import C50Model
from .trees.chaid import CHAIDModel
from .trees.decision_stump import DecisionStumpModel
from .trees.conditional_tree import ConditionalDecisionTreeModel
from .trees.m5 import M5Model

__all__ = [
    "DecisionTreeModel",
    "RandomForestModel",
    "CARTModel",
    "ID3Model",
    "C45Model",
    "C50Model",
    "CHAIDModel",
    "DecisionStumpModel",
    "ConditionalDecisionTreeModel",
    "M5Model",
]
