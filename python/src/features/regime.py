from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from typing import Any


class MarketRegimeDetector:
    """
    Detects market regimes (e.g., Bull, Bear, Sideways) using unsupervised clustering.
    Defaults to Gaussian Mixture Models (GMM) if HMM is unavailable.
    """

    def __init__(self, n_regimes: int = 3, random_state: int = 42):
        self.n_regimes = n_regimes
        self.model = GaussianMixture(
            n_components=n_regimes, 
            random_state=random_state,
            covariance_type="full"
        )
        self.is_fitted = False

    def fit(self, features: np.ndarray | pd.DataFrame) -> MarketRegimeDetector:
        """Fit the clustering model on historical features."""
        if isinstance(features, pd.DataFrame):
            features = features.values
            
        # Remove NaNs
        features = features[~np.isnan(features).any(axis=1)]
        
        self.model.fit(features)
        self.is_fitted = True
        return self

    def predict_regime(self, features: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Predict the current regime for the given features."""
        if not self.is_fitted:
            raise RuntimeError("MarketRegimeDetector must be fitted before prediction.")
            
        if isinstance(features, pd.DataFrame):
            features = features.values
            
        # Handle cases with NaNs by returning a default/previous regime or 0
        if np.isnan(features).any():
            return np.zeros(len(features), dtype=int)
            
        return self.model.predict(features)

    def get_regime_one_hot(self, features: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Return one-hot encoded regime labels."""
        labels = self.predict_regime(features)
        one_hot = np.zeros((len(labels), self.n_regimes))
        one_hot[np.arange(len(labels)), labels] = 1
        return one_hot
