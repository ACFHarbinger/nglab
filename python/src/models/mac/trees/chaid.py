"""CHAID Model."""

from .decision_tree import DecisionTreeModel


class CHAIDModel(DecisionTreeModel):
    """
    Chi-squared Automatic Interaction Detection (CHAID).
    Uses Chi-square tests for splits.
    sklearn doesn't support Chi-square splits natively.
    This is a placeholder wrapper that uses the standard DecisionTreeModel.
    """

    pass
