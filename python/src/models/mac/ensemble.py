"""
Ensemble methods (Bagging, Stacking, Voting).
"""

from sklearn.ensemble import (
    BaggingClassifier,
    BaggingRegressor,
    StackingClassifier,
    StackingRegressor,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from .base import ClassicalModel


class BaggingModel(ClassicalModel):
    def __init__(self, task="regression", **kwargs):
        super().__init__()
        if task == "regression":
            self.model = BaggingRegressor(**kwargs)
        else:
            self.model = BaggingClassifier(**kwargs)


class StackingModel(ClassicalModel):
    """
    Stacked Generalization.
    Requires 'estimators' list of (name, estimator) tuples in kwargs, 
    or defaults to simple Linear+Tree stack.
    """
    def __init__(self, task="regression", **kwargs):
        super().__init__()
        
        # Default estimators if not provided
        if "estimators" not in kwargs:
            if task == "regression":
                kwargs["estimators"] = [
                    ('lr', LinearRegression()),
                    ('tree', DecisionTreeRegressor(max_depth=5))
                ]
                kwargs.setdefault("final_estimator", LinearRegression())
            else:
                kwargs["estimators"] = [
                    ('lr', LogisticRegression()),
                    ('tree', DecisionTreeClassifier(max_depth=5))
                ]
                kwargs.setdefault("final_estimator", LogisticRegression())

        if task == "regression":
            self.model = StackingRegressor(**kwargs)
        else:
            self.model = StackingClassifier(**kwargs)


class VotingModel(ClassicalModel):
    """
    Voting Ensemble (Soft/Hard Voting or Weighted Average).
    """
    def __init__(self, task="regression", **kwargs):
        super().__init__()
        
        if "estimators" not in kwargs:
             if task == "regression":
                kwargs["estimators"] = [
                    ('lr', LinearRegression()),
                    ('tree', DecisionTreeRegressor(max_depth=5))
                ]
             else:
                kwargs["estimators"] = [
                    ('lr', LogisticRegression()),
                    ('tree', DecisionTreeClassifier(max_depth=5))
                ]
        
        if task == "regression":
            self.model = VotingRegressor(**kwargs)
        else:
            # voting='hard' is default, 'soft' for probabilities
            self.model = VotingClassifier(**kwargs)


class WeightedAverageModel(VotingModel):
    """
    Weighted Average (Blending).
    Alias for VotingModel.
    """
    pass
