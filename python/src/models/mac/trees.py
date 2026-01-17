"""
Tree-based models.
"""

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
import numpy as np
import torch

from .base import ClassicalModel


class DecisionTreeModel(ClassicalModel):
    def __init__(self, task="regression", **kwargs):
        super().__init__()
        if task == "regression":
            self.model = DecisionTreeRegressor(**kwargs)
        else:
            self.model = DecisionTreeClassifier(**kwargs)


class RandomForestModel(ClassicalModel):
    def __init__(self, task="regression", **kwargs):
        super().__init__()
        if task == "regression":
            self.model = RandomForestRegressor(**kwargs)
        else:
            self.model = RandomForestClassifier(**kwargs)


class CARTModel(DecisionTreeModel):
    """Classification and Regression Tree (CART). Default sklearn behavior."""
    pass


class ID3Model(DecisionTreeModel):
    """
    Iterative Dichotomiser 3 (ID3).
    Approximated using DecisionTreeClassifier with criterion='entropy'.
    sklearn uses CART by default (binary splits), but entropy criterion mimics ID3 logic.
    """
    def __init__(self, task="classification", **kwargs):
        # ID3 is typically for classification
        if task == "regression":
            # Just use default decision tree for regression if requested
             super().__init__(task=task, **kwargs)
        else:
            # Enforce entropy for ID3-like behavior
            kwargs["criterion"] = "entropy"
            super().__init__(task=task, **kwargs)


class C45Model(DecisionTreeModel):
    """
    C4.5 Algorithm.
    Improved ID3 with support for continuous attributes and pruning.
    Approximated in sklearn with criterion='entropy' and potentially pruning params (ccp_alpha).
    """
    def __init__(self, task="classification", **kwargs):
        if task == "classification":
            kwargs["criterion"] = "entropy"
        super().__init__(task=task, **kwargs)


class C50Model(DecisionTreeModel):
    """
    C5.0 Algorithm.
    Proprietary improvement over C4.5 (faster, smaller trees).
    We approximate this by tuning the underlying CART implementation.
    """
    def __init__(self, task="classification", **kwargs):
        # C5.0 typically implies boosting, but as a single tree model we just use entropy
        # and maybe different pruning.
        if task == "classification":
            kwargs["criterion"] = "entropy"
        super().__init__(task=task, **kwargs)


class CHAIDModel(DecisionTreeModel):
    """
    Chi-squared Automatic Interaction Detection (CHAID).
    Uses Chi-square tests for splits.
    sklearn doesn't support Chi-square splits natively.
    This is a placeholder wrapper that uses the standard DecisionTreeModel.
    """
    pass


class DecisionStumpModel(DecisionTreeModel):
    """Decision Stump: A Decision Tree with max_depth=1."""
    def __init__(self, task="regression", **kwargs):
        kwargs["max_depth"] = 1
        super().__init__(task=task, **kwargs)


class ConditionalDecisionTreeModel(DecisionTreeModel):
    """
    Conditional Decision Tree.
    Approximated by requiring a minimum impurity decrease for splits,
    simulating significance testing.
    """
    def __init__(self, task="regression", min_impurity_decrease=0.05, **kwargs):
        kwargs["min_impurity_decrease"] = min_impurity_decrease
        super().__init__(task=task, **kwargs)


class M5Model(ClassicalModel):
    """
    M5 Algorithm for Regression.
    Builds a tree and then fits linear regression models at the leaves.
    """
    def __init__(self, **kwargs):
        super().__init__()
        # We start with a regression tree
        self.tree = DecisionTreeRegressor(**kwargs)
        self.leaf_models = {}

    def fit(self, X, y):
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
            
        if X.ndim == 3:
            X = X.reshape(X.shape[0] * X.shape[1], -1)
            y = y
            if y.ndim > 1:
                y = y.reshape(y.shape[0] * y.shape[1], -1)
            
        self.tree.fit(X, y)
        
        # Identify leaves and fit linear models
        leaves = self.tree.apply(X)
        unique_leaves = np.unique(leaves)
        
        for leaf in unique_leaves:
            # Get samples in this leaf
            mask = (leaves == leaf)
            X_leaf = X[mask]
            
            # Use original y shape semantics from reshaped y
            if y.ndim == 1:
                y_leaf = y[mask]
            else:
                y_leaf = y[mask] # multi-output
            
            # Simple Linear Regression for the leaf
            if len(X_leaf) > X.shape[1] + 1: # Sufficient samples
                 model = LinearRegression()
                 model.fit(X_leaf, y_leaf)
                 self.leaf_models[leaf] = model
            else:
                 self.leaf_models[leaf] = None

        self._is_fitted = True

    def predict(self, X):
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
            
        if not self._is_fitted:
            return np.zeros((X.shape[0], 1))
        
        # Predict using tree to find leaves
        leaves = self.tree.apply(X)
        pred = np.zeros((X.shape[0], 1) if self.tree.n_outputs_ == 1 else (X.shape[0], self.tree.n_outputs_))
        tree_pred = self.tree.predict(X) 
        if tree_pred.ndim == 1:
            tree_pred = tree_pred[:, np.newaxis]
        
        for i, leaf in enumerate(leaves):
            lm = self.leaf_models.get(leaf)
            if lm:
                p = lm.predict(X[i:i+1])
                pred[i] = p if p.ndim == 1 else p[0]
            else:
                pred[i] = tree_pred[i]
                
        return pred

    def forward(self, x, **kwargs):
        device = x.device
        x_np = x.detach().cpu().numpy()
        if x_np.ndim == 3:
             x_np = x_np[:, -1, :]
             
        out_np = self.predict(x_np)
        return torch.from_numpy(out_np).to(device).to(torch.float32)
