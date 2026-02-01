import numpy as np
import pandas as pd
import pytest

from python.src.features.pipeline import FeaturePipeline


@pytest.fixture
def lob_sample_data():
    n_samples = 100
    dates = pd.date_range("2024-01-01", periods=n_samples, freq="min")
    
    # Basic OHLCV
    close = 100 + np.cumsum(np.random.normal(0, 1, n_samples))
    volume = np.random.randint(100, 1000, n_samples)
    
    # LOB data
    bid_p0 = close - 0.1
    ask_p0 = close + 0.1
    bid_v0 = np.random.randint(10, 100, n_samples)
    ask_v0 = np.random.randint(10, 100, n_samples)
    
    df = pd.DataFrame({
        "close": close,
        "volume": volume,
        "bid_p0": bid_p0,
        "ask_p0": ask_p0,
        "bid_v0": bid_v0,
        "ask_v0": ask_v0,
    }, index=dates)
    return df

def test_pipeline_lob_features(lob_sample_data):
    pipe = FeaturePipeline(lookback=10)
    pipe.fit(lob_sample_data)
    
    output = pipe.transform(lob_sample_data)
    
    # Check if LOB features are present in feature_names and output
    assert "imbalance" in pipe.feature_names
    assert "spread" in pipe.feature_names
    assert "vwap" in pipe.feature_names
    
    # Regime features should also be there (defaults to 3 regimes)
    assert any("regime_" in name for name in pipe.feature_names)
    assert "regime_0" in pipe.feature_names
    
    assert output.shape[1] == len(pipe.feature_names)

def test_pipeline_no_lob_features():
    # Test that it still works without LOB data
    df = pd.DataFrame({
        "close": np.random.normal(100, 1, 50),
        "volume": np.random.normal(1000, 10, 50)
    })
    pipe = FeaturePipeline(lookback=5)
    pipe.fit(df)
    
    output = pipe.transform(df)
    
    assert "imbalance" not in pipe.feature_names
    assert any("regime_" in name for name in pipe.feature_names) # Regimes should still exist for technical indicators
    assert output.shape[1] == len(pipe.feature_names)
