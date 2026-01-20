"""Tree-based models package."""

from .c45 import C45Model
from .c50 import C50Model
from .cart import CARTModel
from .chaid import CHAIDModel
from .conditional_tree import ConditionalDecisionTreeModel
from .decision_stump import DecisionStumpModel
from .decision_tree import DecisionTreeModel
from .id3 import ID3Model
from .m5 import M5Model
from .random_forest import RandomForestModel

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
