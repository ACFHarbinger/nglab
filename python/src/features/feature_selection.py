"""
Automated Feature Selection Toolkit for Time Series Data.
"""

from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from numpy.typing import NDArray
from sklearn.base import BaseEstimator
from sklearn.feature_selection import RFECV, mutual_info_regression
from sklearn.model_selection import TimeSeriesSplit


class TimeSeriesFeatureSelector:
    """
    Toolkit for selecting the most relevant features in time series forecasting.
    """

    @staticmethod
    def compute_mutual_info(
        X: pd.DataFrame,  # noqa: N803
        y: pd.Series,
        discrete_features: bool | list[int] = False,
    ) -> pd.Series:
        """
        Compute Mutual Information between features and target.
        Handles non-linear dependencies unlike simple correlation.
        """
        # Ensure no NaNs as MI doesn't like them
        X_clean = X.fillna(0)
        y_clean = y.fillna(0)

        mi_scores = mutual_info_regression(
            X_clean, y_clean, discrete_features=discrete_features
        )
        mi_series = pd.Series(mi_scores, index=X.columns)
        return mi_series.sort_values(ascending=False)

    @staticmethod
    def run_rfecv(  # noqa: PLR0913
        estimator: BaseEstimator,
        X: pd.DataFrame,  # noqa: N803
        y: pd.Series,
        step: int = 1,
        cv: int = 5,
        scoring: str = "neg_mean_absolute_error",
    ) -> tuple[RFECV, list[str]]:
        """
        Recursive Feature Elimination with Cross-Validation.
        Automatically finds the optimal number of features.
        """
        cv_split = TimeSeriesSplit(n_splits=cv)
        selector = RFECV(
            estimator=estimator, step=step, cv=cv_split, scoring=scoring, n_jobs=-1
        )
        selector.fit(X, y)

        selected_features: list[str] = list(X.columns[selector.support_])
        return selector, selected_features

    @staticmethod
    def plot_importance(scores: pd.Series, title: str = "Feature Importance") -> None:
        """Visualize feature importance scores."""
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=scores.values, y=scores.index)
        ax.set_title(title)
        ax.set_xlabel("Score")
        ax.set_ylabel("Features")
        fig.tight_layout()
        plt.show()


HAS_SHAP = False
try:
    import shap

    HAS_SHAP = True
except ImportError:
    shap = None


class SHAPToolkit:
    """Wrapper for SHAP (SHapley Additive exPlanations)."""

    def __init__(self, model: Any, background_data: NDArray[Any] | None = None) -> None:
        """
        Initialize SHAP toolkit.

        Args:
            model: Trained model to explain.
            background_data: Representative dataset for the explainer.
        """
        self.model = model
        self.background_data = background_data
        self.explainer: Any = None

        if not HAS_SHAP:
            print("Warning: SHAP library not found. SHAPToolkit will be limited.")

    def explain(self, X: NDArray[Any]) -> NDArray[Any]:  # noqa: N803
        """Calculate SHAP values for the given data."""
        if not HAS_SHAP or shap is None:
            raise RuntimeError("SHAP not installed. Install with 'pip install shap'")

        if self.explainer is None:
            self.explainer = shap.Explainer(self.model, self.background_data)

        return self.explainer(X)  # type: ignore

    def plot_summary(self, shap_values: Any, X: pd.DataFrame) -> None:  # noqa: N803
        """Plot SHAP summary plot."""
        if not HAS_SHAP or shap is None:
            return

        shap.summary_plot(shap_values, X)
