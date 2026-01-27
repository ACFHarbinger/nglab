"""
Automated Feature Engineering Pipeline.

Provides a robust, serializable pipeline for transforming raw market data
into model-ready features, leveraging GPU acceleration where possible.
"""

from typing import Any, cast

import joblib
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
        """Initialize FeaturePipeline."""
        self.lookback = lookback
        self.feature_set = feature_set
        self.scaler_type = scaler_type
        self.gpu_device = gpu_device
        self.selection_threshold = selection_threshold
        self.selection_method = selection_method
        self.selection_params = selection_params or {}

        # Components
        from python.src.features.regime import MarketRegimeDetector
        self.regime_detector = MarketRegimeDetector(n_regimes=self.selection_params.get("n_regimes", 3))

        # Components
        # Components
        self.gpu_engineer: Any = None
        self.scaler: Any = None
        self.selector: Any = None
        self.feature_names: list[str] = []

    def fit(
        self, x: pd.DataFrame | np.ndarray[Any, Any], y: Any = None
    ) -> "FeaturePipeline":
        """
        Fit the pipeline components (e.g., scalers) on historical data.

        Args:
            X: Input DataFrame with 'close', 'high', 'low', 'volume' columns.
        """
        # 1. Generate features temporarily to fit scaler
        features = self._generate_features(x)
        features_clean = features.dropna()

        # 2. Fit Regime Detector
        self.regime_detector.fit(features_clean)
        regime_one_hot = self.regime_detector.get_regime_one_hot(features_clean)
        
        # Combine base features and regimes
        regime_df = pd.DataFrame(
            regime_one_hot, 
            index=features_clean.index, 
            columns=[f"regime_{i}" for i in range(regime_one_hot.shape[1])]
        )
        full_features = pd.concat([features_clean, regime_df], axis=1)

        # 3. Fit Scaler
        if self.scaler_type == "standard":
            self.scaler = StandardScaler()
        elif self.scaler_type == "robust":
            self.scaler = RobustScaler()
        elif self.scaler_type == "online":
            from python.src.features.normalization import OnlineNormalizer
            self.scaler = OnlineNormalizer(feature_dim=full_features.shape[1])
        else:
            raise ValueError(f"Unknown scaler type: {self.scaler_type}")

        # Clean NaNs before fitting scaler
        self.scaler.fit(full_features)

        # 4. Fit Selector
        if self.selection_method == "variance":
            from sklearn.feature_selection import VarianceThreshold

            self.selector = VarianceThreshold(threshold=self.selection_threshold)
            self.selector.fit(full_features)
        elif self.selection_method == "mi":
            from sklearn.feature_selection import SelectKBest, mutual_info_regression
            from python.src.features.feature_selection import TimeSeriesFeatureSelector

            self.selector = SelectKBest(
                mutual_info_regression, k=self.selection_params.get("n_features", 10)
            )
            self.selector.fit(
                full_features, y if y is not None else full_features.iloc[:, 0]
            )
        elif self.selection_method == "rfecv":
            from sklearn.ensemble import RandomForestRegressor
            from python.src.features.feature_selection import TimeSeriesFeatureSelector

            estimator = self.selection_params.get(
                "estimator", RandomForestRegressor(n_estimators=10, n_jobs=-1)
            )
            self.selector, _ = TimeSeriesFeatureSelector.run_rfecv(
                estimator,
                full_features,
                y if y is not None else full_features.iloc[:, 0],
                step=self.selection_params.get("step", 1),
                cv=self.selection_params.get("cv", 3),
            )
        else:
            raise ValueError(f"Unknown selection method: {self.selection_method}")

        # Update feature names
        if self.selector:
            selected_mask = self.selector.get_support()
            if isinstance(full_features, pd.DataFrame):
                self.feature_names = [
                    str(full_features.columns[i])
                    for i, selected in enumerate(selected_mask)
                    if selected
                ]
            else:
                self.feature_names = [
                    f"feat_{i}" for i, selected in enumerate(selected_mask) if selected
                ]

        return self

    def transform(self, x: pd.DataFrame | np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """
        Transform new data into model inputs.
        """
        features = self._generate_features(x)

        # Handle recent NaNs (fill with 0 or forward fill)
        if hasattr(features, "ffill"):
            features_filled = features.ffill().fillna(0.0)
        else:
            features_filled = pd.DataFrame(features).ffill().fillna(0.0)

        # Add Regimes
        regime_one_hot = self.regime_detector.get_regime_one_hot(features_filled)
        regime_df = pd.DataFrame(
            regime_one_hot, 
            index=features_filled.index, 
            columns=[f"regime_{i}" for i in range(regime_one_hot.shape[1])]
        )
        full_features = pd.concat([features_filled, regime_df], axis=1)

        if self.scaler:
            scaled = self.scaler.transform(full_features)
        else:
            raise RuntimeError("Pipeline must be fitted before transform")

        if self.selector:
            scaled = self.selector.transform(scaled)

        return cast(np.ndarray[Any, Any], scaled)

    def _generate_features(
        self,
        x: pd.DataFrame | np.ndarray[Any, Any],
    ) -> pd.DataFrame:
        """
        Internal method to generate raw features using GPU acceleration.
        """
        # Ensure DataFrame
        if isinstance(x, np.ndarray):
            # Assume single column 'close' if 1D, else specific order
            if x.ndim == 1:
                df = pd.DataFrame({"close": x})
            else:
                # Minimal expected columns
                cols = ["close", "volume", "high", "low", "open"]
                df = pd.DataFrame(x, columns=cols[: x.shape[1]])
        else:
            df = x.copy()

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

        # 5. LOB Features (if present)
        # Expected columns: bid_p0, ask_p0, bid_v0, ask_v0
        lob_features = pd.DataFrame(index=df.index)
        if all(c in df.columns for c in ["bid_p0", "ask_p0", "bid_v0", "ask_v0"]):
            bid_v = torch.tensor(df["bid_v0"].values, dtype=torch.float32)
            ask_v = torch.tensor(df["ask_v0"].values, dtype=torch.float32)
            bid_p = torch.tensor(df["bid_p0"].values, dtype=torch.float32)
            ask_p = torch.tensor(df["ask_p0"].values, dtype=torch.float32)
            
            lob_features["imbalance"] = self.gpu_engineer.compute_imbalance(bid_v, ask_v).cpu().numpy()
            lob_features["spread"] = self.gpu_engineer.compute_spread(bid_p, ask_p).cpu().numpy()
            lob_features["vwap"] = self.gpu_engineer.compute_vwap(close_tensor, df["volume"].values if "volume" in df.columns else bid_v).cpu().numpy()

        # Move back to CPU/Pandas for alignment
        # (For pure GPU pipeline we'd stay in tensor, but we need sklearn for now)
        features = pd.DataFrame(index=df.index)
        features["log_ret"] = log_ret.cpu().numpy()
        features["sma_diff"] = (sma_short - sma_long).cpu().numpy()
        features["rsi"] = rsi.cpu().numpy()
        features["volatility"] = (
            df["close"].rolling(window=20).std().fillna(0)
        )  # CPU fallback for now if not in GPU lib

        if not lob_features.empty:
            features = pd.concat([features, lob_features], axis=1)

        return features

    def save(self, path: str) -> None:
        """Save pipeline state."""
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "FeaturePipeline":
        """Load pipeline state."""
        return joblib.load(path)  # type: ignore
