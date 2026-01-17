"""M5 Model."""

import numpy as np
import torch
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from ..base import ClassicalModel


class M5Model(ClassicalModel):
    """
    M5 Algorithm for Regression.
    Builds a tree and then fits linear regression models at the leaves.
    """
    def __init__(self, **kwargs):
        super().__init__()
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
        
        leaves = self.tree.apply(X)
        unique_leaves = np.unique(leaves)
        
        for leaf in unique_leaves:
            mask = (leaves == leaf)
            X_leaf = X[mask]
            
            if y.ndim == 1:
                y_leaf = y[mask]
            else:
                y_leaf = y[mask]
            
            if len(X_leaf) > X.shape[1] + 1:
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
