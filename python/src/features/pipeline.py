"""
Automated Feature Engineering Pipeline.

Provides a robust, serializable pipeline for transforming raw market data
into model-ready features, leveraging GPU acceleration where possible.
"""

from typing import List, Optional, Union, Dict, Any
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, RobustScaler

from python.src.utils.functions.gpu_features import GPUFeatureEngineer


class FeaturePipeline(BaseEstimator, TransformerMixin):
    """
    End-to-end feature engineering pipeline.

    Stages:
    1. Validation & Cleaning
    2. GPU Feature Generation (Technical Indicators)
    3. Normalization/Scaling (CPU-based using sklearn for state management)
    4. Feature Selection (VarianceThreshold)
    """

    def __init__(
        self,
        lookback: int = 30,
        feature_set: str = "standard",
        scaler_type: str = "robust",
        gpu_device: Optional[str] = None,
        selection_threshold: float = 0.0,
    ):
        self.lookback = lookback
        self.feature_set = feature_set
        self.scaler_type = scaler_type
        self.gpu_device = gpu_device
        self.selection_threshold = selection_threshold

        # Components
        self.gpu_engineer = None
        self.scaler = None
        self.selector = None
        self.feature_names: List[str] = []

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None):
        """
        Fit the pipeline components (e.g., scalers) on historical data.

        Args:
            X: Input DataFrame with 'close', 'high', 'low', 'volume' columns.
        """
        # 1. Generate features temporarily to fit scaler
        features = self._generate_features(X)

        # 2. Fit Scaler
        if self.scaler_type == "standard":
            self.scaler = StandardScaler()
        elif self.scaler_type == "robust":
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaler type: {self.scaler_type}")

        # Clean NaNs before fitting scaler
        # (Drop initial rows required for lookback)
        features_clean = features.dropna()
        self.scaler.fit(features_clean)

        # 3. Fit Selector
        from sklearn.feature_selection import VarianceThreshold

        self.selector = VarianceThreshold(threshold=self.selection_threshold)
        self.selector.fit(features_clean)

        # Update feature names
        selected_mask = self.selector.get_support()
        self.feature_names = features.columns[selected_mask].tolist()

        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Transform new data into model inputs.
        """
        features = self._generate_features(X)

        # Handle recent NaNs (fill with 0 or forward fill)
        # For production inference, we usually have a buffer history.
        features_filled = features.fillna(method="ffill").fillna(0.0)

        if self.scaler:
            scaled = self.scaler.transform(features_filled)
        else:
            # Should reset/fit if not trained, but transform shouldn't fit
            raise RuntimeError("Pipeline must be fitted before transform")

        if self.selector:
            scaled = self.selector.transform(scaled)

        return scaled

    def _generate_features(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """
        Internal method to generate raw features using GPU acceleration.
        """
        # Ensure DataFrame
        if isinstance(X, np.ndarray):
            # Assume single column 'close' if 1D, else specific order
            if X.ndim == 1:
                df = pd.DataFrame({"close": X})
            else:
                # Minimal expected columns
                df = pd.DataFrame(X, columns=["close", "volume"][: X.shape[1]])
        else:
            df = X.copy()

        # Initialize GPU Engineer on demand (to avoid serialization issues)
        if self.gpu_engineer is None:
            self.gpu_engineer = GPUFeatureEngineer(device=self.gpu_device)

        # Convert to tensor
        close_tensor = torch.tensor(df["close"].values, dtype=torch.float32)

        # --- GPU Feature Generation ---
        # 1. Returns
        log_ret = torch.log(close_tensor / close_tensor.roll(1))
        log_ret[0] = 0

        # 2. SMA
        sma_short = self.gpu_engineer.moving_average(close_tensor, window=10)
        sma_long = self.gpu_engineer.moving_average(close_tensor, window=self.lookback)

        # 3. RSI
        rsi = self.gpu_engineer.rsi(close_tensor, window=14)

        # 4. Bollinger Bands
        # upper, mid, lower = self.gpu_engineer.bollinger_bands(close_tensor, window=20)

        # Move back to CPU/Pandas for alignment
        # (For pure GPU pipeline we'd stay in tensor, but we need sklearn for now)
        features = pd.DataFrame(index=df.index)
        features["log_ret"] = log_ret.cpu().numpy()
        features["sma_diff"] = (sma_short - sma_long).cpu().numpy()
        features["rsi"] = rsi.cpu().numpy()
        features["volatility"] = (
            df["close"].rolling(window=20).std().fillna(0)
        )  # CPU fallback for now if not in GPU lib

        # Add basic time features if available
        # if 'timestamp' in df.columns: ...

        return features

    def save(self, path: str):
        """Save pipeline state."""
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "FeaturePipeline":
        """Load pipeline state."""
        return joblib.load(path)
