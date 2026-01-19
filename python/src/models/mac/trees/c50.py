"""C5.0 Model."""

from .decision_tree import DecisionTreeModel


class C50Model(DecisionTreeModel):
    """
    C5.0 Algorithm.
    Proprietary improvement over C4.5 (faster, smaller trees).
    """

    def __init__(self, task="classification", **kwargs):
        if task == "classification":
            kwargs["criterion"] = "entropy"
        super().__init__(task=task, **kwargs)
