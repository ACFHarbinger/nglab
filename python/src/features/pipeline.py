"""
Automated Feature Engineering Pipeline.

Provides a robust, serializable pipeline for transforming raw market data
into model-ready features, leveraging GPU acceleration where possible.
"""

from typing import Any, cast

import joblib  # type: ignore
import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler, StandardScaler

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

    def __init__(  # noqa: PLR0913
        self,
        lookback: int = 30,
        feature_set: str = "standard",
        scaler_type: str = "robust",
        gpu_device: str | None = None,
        selection_threshold: float = 0.0,
        selection_method: str = "variance",  # "variance", "mi", "rfecv"
        selection_params: dict[str, Any] | None = None,
    ):
        self.lookback = lookback
        self.feature_set = feature_set
        self.scaler_type = scaler_type
        self.gpu_device = gpu_device
        self.selection_threshold = selection_threshold
        self.selection_method = selection_method
        self.selection_params = selection_params or {}

        # Components
        # Components
        self.gpu_engineer: Any = None
        self.scaler: Any = None
        self.selector: Any = None
        self.feature_names: list[str] = []

    def fit(self, X: pd.DataFrame | np.ndarray[Any, Any], y: Any = None) -> "FeaturePipeline":
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
        if self.selection_method == "variance":
            from sklearn.feature_selection import VarianceThreshold

            self.selector = VarianceThreshold(threshold=self.selection_threshold)
            self.selector.fit(features_clean)
        elif self.selection_method == "mi":
            from python.src.utils.feature_selection import TimeSeriesFeatureSelector

            mi_scores = TimeSeriesFeatureSelector.compute_mutual_info(
                features_clean, y if y is not None else features_clean.iloc[:, 0]
            )
            mi_scores.head(self.selection_params.get("n_features", 10)).index.tolist()

            # Create a simple PassThrough selector that just picks columns
            from sklearn.ensemble import RandomForestRegressor

            # We use SelectFromModel with a dummy if needed, but easier to just use a custom one
            # For simplicity, we can use SelectKBest with MI
            from sklearn.feature_selection import SelectKBest, mutual_info_regression

            self.selector = SelectKBest(
                mutual_info_regression, k=self.selection_params.get("n_features", 10)
            )
            self.selector.fit(
                features_clean, y if y is not None else features_clean.iloc[:, 0]
            )
        elif self.selection_method == "rfecv":
            from sklearn.ensemble import RandomForestRegressor

            from python.src.utils.feature_selection import TimeSeriesFeatureSelector

            estimator = self.selection_params.get(
                "estimator", RandomForestRegressor(n_estimators=10, n_jobs=-1)
            )
            self.selector, _ = TimeSeriesFeatureSelector.run_rfecv(
                estimator,
                features_clean,
                y if y is not None else features_clean.iloc[:, 0],
                step=self.selection_params.get("step", 1),
                cv=self.selection_params.get("cv", 3),
            )
        else:
            raise ValueError(f"Unknown selection method: {self.selection_method}")

        # Update feature names
        if self.selector:
            selected_mask = self.selector.get_support()
            if isinstance(features, pd.DataFrame):
                self.feature_names = [
                    str(features.columns[i])
                    for i, selected in enumerate(selected_mask)
                    if selected
                ]
            else:
                self.feature_names = [
                    f"feat_{i}" for i, selected in enumerate(selected_mask) if selected
                ]

        return self

    def transform(self, X: pd.DataFrame | np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:  # noqa: N803
        """
        Transform new data into model inputs.
        """
        features = self._generate_features(X)

        # Handle recent NaNs (fill with 0 or forward fill)
        # For production inference, we usually have a buffer history.
        if hasattr(features, "ffill"):
            features_filled = features.ffill().fillna(0.0)
        else:
            features_filled = pd.DataFrame(features).ffill().fillna(0.0)

        if self.scaler:
            scaled = self.scaler.transform(features_filled)
        else:
            # Should reset/fit if not trained, but transform shouldn't fit
            raise RuntimeError("Pipeline must be fitted before transform")

        if self.selector:
            scaled = self.selector.transform(scaled)

        return cast(np.ndarray[Any, Any], scaled)

    def _generate_features(
        self,
        X: pd.DataFrame | np.ndarray[Any, Any],  # noqa: N803
    ) -> pd.DataFrame:
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
                cols = ["close", "volume", "high", "low", "open"]
                df = pd.DataFrame(X, columns=cols[: X.shape[1]])
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

    def save(self, path: str) -> None:
        """Save pipeline state."""
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "FeaturePipeline":
        """Load pipeline state."""
        return joblib.load(path)  # type: ignore
