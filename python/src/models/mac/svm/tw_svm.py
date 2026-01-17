"""Twin SVM Model."""

import numpy as np
import torch
from sklearn.svm import LinearSVC
from ..base import ClassicalModel


class TWSVMModel(ClassicalModel):
    """
    Twin Support Vector Machine (TWSVM).
    Simplified implementation for binary classification.
    """
    def __init__(self, c1=1.0, c2=1.0, epsilon=1e-5, **kwargs):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.epsilon = epsilon
        self.weights1 = None
        self.weights2 = None
        self.bias1 = None
        self.bias2 = None

    def fit(self, X, y):
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
            
        if X.ndim == 3:
            X = X.reshape(X.shape[0] * X.shape[1], -1)
            y = y.reshape(y.shape[0] * y.shape[1], -1)

        y = y.ravel()
        classes = np.unique(y)
        if len(classes) != 2:
            self.fallback = LinearSVC(C=self.c1)
            self.fallback.fit(X, y)
            self._is_fitted = True
            return
        
        self.fallback = None
        
        A = X[y == classes[0]]
        B = X[y == classes[1]]
        
        m1 = A.shape[0]
        m2 = B.shape[0]
        e1 = np.ones((m1, 1))
        e2 = np.ones((m2, 1))
        
        H = np.hstack((A, e1))
        G = np.hstack((B, e2))
        
        X_full = np.vstack((H, G))
        y_1 = np.vstack((np.zeros((m1, 1)), np.ones((m2, 1))))
        
        pseudo_inv1 = np.linalg.pinv(X_full.T @ X_full + self.epsilon * np.eye(X_full.shape[1]))
        z1 = pseudo_inv1 @ X_full.T @ y_1
        self.weights1 = z1[:-1]
        self.bias1 = z1[-1]
        
        X_full2 = np.vstack((G, H))
        y_2 = np.vstack((np.zeros((m2, 1)), np.ones((m1, 1))))
        
        pseudo_inv2 = np.linalg.pinv(X_full2.T @ X_full2 + self.epsilon * np.eye(X_full2.shape[1]))
        z2 = pseudo_inv2 @ X_full2.T @ y_2
        self.weights2 = z2[:-1]
        self.bias2 = z2[-1]
        
        self.classes_ = classes
        self._is_fitted = True

    def predict(self, X):
        if not self._is_fitted:
            return np.zeros((X.shape[0], 1))
            
        if self.fallback:
            return self.fallback.predict(X).reshape(-1, 1)

        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
            
        dist1 = np.abs(X @ self.weights1 + self.bias1)
        dist2 = np.abs(X @ self.weights2 + self.bias2)
        
        preds_idx = (dist1 > dist2).astype(int).ravel()
        return self.classes_[preds_idx].reshape(-1, 1)

    def forward(self, x, **kwargs):
        device = x.device
        x_np = x.detach().cpu().numpy()
        if x_np.ndim == 3:
            x_np = x_np[:, -1, :] 
        
        out_np = self.predict(x_np)
        return torch.from_numpy(out_np).to(device).to(torch.float32)
