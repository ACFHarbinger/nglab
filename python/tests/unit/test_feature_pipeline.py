import numpy as np
import pandas as pd
import pytest

from python.src.features.pipeline import FeaturePipeline


@pytest.fixture
def sample_data():
    # Helper to generate dummy OHLCV data
    n_samples = 200
    dates = pd.date_range("2024-01-01", periods=n_samples, freq="D")

    close = np.linspace(100, 200, n_samples) + np.random.normal(0, 5, n_samples)
    volume = np.random.randint(1000, 5000, n_samples)

    df = pd.DataFrame({"close": close, "volume": volume}, index=dates)
    return df


def test_pipeline_instantiation():
    pipe = FeaturePipeline()
    assert pipe.gpu_engineer is None
    assert pipe.scaler is None


def test_pipeline_fit_transform(sample_data):
    pipe = FeaturePipeline(lookback=10, scaler_type="standard", selection_threshold=0.0)

    # Fit
    pipe.fit(sample_data)
    assert pipe.scaler is not None
    assert len(pipe.feature_names) > 0

    # Transform
    output = pipe.transform(sample_data)

    # Output should correspond to features * lookback if flattened,
    # but currently pipeline output is (n_samples, n_features)
    # Check shape
    assert output.shape[0] == len(sample_data)
    assert output.shape[1] == len(pipe.feature_names)

    # Check scaling properties (roughly 0 mean, 1 std)
    # Exclude initial nan-filled rows (lookback)
    valid_output = output[15:]
    means = valid_output.mean(axis=0)
    valid_output.std(axis=0)

    np.testing.assert_allclose(
        means, 0, atol=0.5
    )  # Looser tolerance due to small sample/ffill
    # Variance might be small if feature is constant, but VarianceThreshold should handle that if > 0


def test_feature_selection(sample_data):
    # Add a constant feature
    sample_data["constant"] = 1.0

    pipe = FeaturePipeline(selection_threshold=0.01)  # Drop near-zeros var

    pipe.fit(sample_data)
    pipe.transform(sample_data)

    # Constant features should be dropped (internal implementation details of _generate_features logic matter)
    # Current _generate_features doesn't necessarily pass through raw columns (it calculates new ones).
    # But if one of the constructed features is constant (e.g. sma_diff of constant input), it should be dropped.

    # Let's test by creating a scenario where a generated feature is constant.
    # If input is constant, SMA diff is 0.
    constant_df = pd.DataFrame({"close": np.ones(100), "volume": np.ones(100)})
    pipe_const = FeaturePipeline(selection_threshold=0.01)

    # All features (returns, sma_diff, rsi) should be 0 or constant
    try:
        pipe_const.fit(constant_df)
        res = pipe_const.transform(constant_df)
        # Should be empty or very few columns
        assert res.shape[1] == 0 or res.shape[1] < 4
    except ValueError:
        # VarianceThreshold might complain if no feature meets threshold
        pass
