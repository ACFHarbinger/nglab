import numpy as np
import pandas as pd
import pytest

from python.src.features.feature_selection import TimeSeriesFeatureSelector
from python.src.features.pipeline import FeaturePipeline


@pytest.fixture
def sample_data():
    """Generate dummy market data with some redundant features."""
    n_points = 100
    prices = np.linspace(100, 110, n_points) + np.random.normal(0, 0.5, n_points)

    df = pd.DataFrame(
        {
            "close": prices,
            "volume": np.random.randint(1000, 5000, n_points),
            "high": prices + 0.5,
            "low": prices - 0.5,
            "open": prices - 0.1,
        }
    )

    # Add redundant/noisy features
    df["noise1"] = np.random.normal(0, 1, n_points)
    df["noise2"] = np.random.normal(0, 1, n_points)

    # Target is future return
    y = pd.Series(df["close"].shift(-1) / df["close"] - 1).fillna(0)

    return df, y


def test_mutual_info_selection(sample_data):
    df, y = sample_data
    # We only take the columns for MI
    X = df.drop(columns=[])

    mi_scores = TimeSeriesFeatureSelector.compute_mutual_info(X, y)

    assert isinstance(mi_scores, pd.Series)
    assert len(mi_scores) == len(X.columns)
    # Basic sanity: price/volume should ideally have more MI than random noise
    # but since it's random, we just check they exist.
    assert "close" in mi_scores.index


def test_feature_pipeline_mi(sample_data):
    df, y = sample_data

    pipeline = FeaturePipeline(
        lookback=10, selection_method="mi", selection_params={"n_features": 3}
    )

    pipeline.fit(df, y)

    assert len(pipeline.feature_names) == 3

    transformed = pipeline.transform(df)
    assert transformed.shape[1] == 3
    assert isinstance(transformed, np.ndarray)


def test_feature_pipeline_rfecv(sample_data):
    df, y = sample_data

    # RFECV is slow, so use few features and small CV
    pipeline = FeaturePipeline(
        lookback=10, selection_method="rfecv", selection_params={"cv": 2, "step": 1}
    )

    pipeline.fit(df, y)

    assert len(pipeline.feature_names) > 0

    transformed = pipeline.transform(df)
    assert transformed.shape[1] == len(pipeline.feature_names)
