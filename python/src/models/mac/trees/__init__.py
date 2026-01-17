"""Tree-based models package."""

from .decision_tree import DecisionTreeModel
from .random_forest import RandomForestModel
from .cart import CARTModel
from .id3 import ID3Model
from .c45 import C45Model
from .c50 import C50Model
from .chaid import CHAIDModel
from .decision_stump import DecisionStumpModel
from .conditional_tree import ConditionalDecisionTreeModel
from .m5 import M5Model

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
