"""
Automated Feature Selection Toolkit for Time Series Data.
"""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator
from sklearn.feature_selection import RFECV, mutual_info_regression


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
        )  # type: ignore
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
        selector = RFECV(
            estimator=estimator, step=step, cv=cv, scoring=scoring, n_jobs=-1
        )
        selector.fit(X, y)

        selected_features = X.columns[selector.support_].tolist()
        return selector, selected_features

    @staticmethod
    def plot_importance(scores: pd.Series, title: str = "Feature Importance"):
        """Visualize feature importance scores."""
        plt.figure(figsize=(10, 6))
        sns.barplot(x=scores.values, y=scores.index)
        plt.title(title)
        plt.xlabel("Score")
        plt.ylabel("Features")
        plt.tight_layout()
        plt.show()


try:
    import shap  # noqa: F401

    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


class SHAPToolkit:
    """Wrapper for SHAP (SHapley Additive exPlanations)."""

    def __init__(self, model: Any, background_data: np.ndarray | None = None):
        self.model = model
        self.background_data = background_data
        self.explainer = None

        if not HAS_SHAP:
            print("Warning: SHAP library not found. SHAPToolkit will be limited.")

    def explain(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
        """Calculate SHAP values for the given data."""
        if not HAS_SHAP:
            raise RuntimeError("SHAP not installed. Install with 'pip install shap'")

        if self.explainer is None:
            # Automatic explainer choice
            import shap as shap_pkg

            self.explainer = shap_pkg.Explainer(self.model, self.background_data)

        return self.explainer(X)

    def plot_summary(self, shap_values: Any, X: pd.DataFrame):  # noqa: N803
        """Plot SHAP summary plot."""
        if not HAS_SHAP:
            return
        import shap as shap_pkg

        shap_pkg.summary_plot(shap_values, X)
