"""C4.5 Model."""

from .decision_tree import DecisionTreeModel


class C45Model(DecisionTreeModel):
    """
    C4.5 Algorithm.
    Improved ID3 with support for continuous attributes and pruning.
    """
    def __init__(self, task="classification", **kwargs):
        if task == "classification":
            kwargs["criterion"] = "entropy"
        super().__init__(task=task, **kwargs)
