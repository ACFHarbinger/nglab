"""ID3 Model."""

from .decision_tree import DecisionTreeModel


class ID3Model(DecisionTreeModel):
    """
    Iterative Dichotomiser 3 (ID3).
    Approximated using DecisionTreeClassifier with criterion='entropy'.
    """

    def __init__(self, task="classification", **kwargs):
        if task == "regression":
            super().__init__(task=task, **kwargs)
        else:
            kwargs["criterion"] = "entropy"
            super().__init__(task=task, **kwargs)
