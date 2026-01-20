"""Tree-based models facade."""

from .trees.c45 import C45Model
from .trees.c50 import C50Model
from .trees.cart import CARTModel
from .trees.chaid import CHAIDModel
from .trees.conditional_tree import ConditionalDecisionTreeModel
from .trees.decision_stump import DecisionStumpModel
from .trees.decision_tree import DecisionTreeModel
from .trees.id3 import ID3Model
from .trees.m5 import M5Model
from .trees.random_forest import RandomForestModel

__all__ = [
    "C45Model",
    "C50Model",
    "CARTModel",
    "CHAIDModel",
    "ConditionalDecisionTreeModel",
    "DecisionStumpModel",
    "DecisionTreeModel",
    "ID3Model",
    "M5Model",
    "RandomForestModel",
]
