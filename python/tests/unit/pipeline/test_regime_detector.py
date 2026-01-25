import numpy as np
import pytest
import torch

from python.src.pipeline.meta.regime_detector import RegimeDetector


@pytest.fixture
def detector():
    return RegimeDetector(n_regimes=3, window_size=10)


@pytest.fixture
def dummy_prices():
    # Create some dummy price data with different "regimes"
    # Regime 0: Low volatility
    low_vol = np.linspace(100, 101, 50) + np.random.normal(0, 0.1, 50)
    # Regime 1: High volatility
    high_vol = np.linspace(101, 105, 50) + np.random.normal(0, 2.0, 50)
    # Regime 2: Trending
    trending = np.linspace(105, 120, 50) + np.random.normal(0, 0.2, 50)

    return np.concatenate([low_vol, high_vol, trending])


def test_regime_detector_init(detector):
    assert detector.n_regimes == 3
    assert detector.window_size == 10
    assert detector.kmeans is None


def test_extract_features(detector):
    prices = np.random.randn(20)
    features = detector.extract_features(prices)
    # len(prices) - window_size = 20 - 10 = 10
    assert features.shape == (10, 3)  # [volatility, trend, range_pct]


def test_fit_predict(detector, dummy_prices):
    detector.fit(dummy_prices)
    assert detector.kmeans is not None

    # Predict on a piece of the same data
    regime = detector.predict(dummy_prices[-20:])
    assert 0 <= regime < 3

    # Check regime name
    name = detector.get_regime_name(regime)
    assert name in ["Volatile", "Trending", "Ranging"]


def test_predict_not_fitted(detector):
    with pytest.raises(ValueError, match="Detector not fitted"):
        detector.predict(np.random.randn(20))


def test_predict_too_few_prices(detector, dummy_prices):
    detector.fit(dummy_prices)
    with pytest.raises(ValueError, match="Need at least 10 prices"):
        detector.predict(np.random.randn(5))


def test_partition_by_regime(detector, dummy_prices):
    # data size should match number of windows
    num_windows = len(dummy_prices) - detector.window_size
    dummy_data = torch.randn(num_windows, 5)

    partitions = detector.partition_by_regime(dummy_prices, dummy_data)

    assert len(partitions) == 3
    total_samples = sum(p.size(0) for p in partitions.values())
    assert total_samples == num_windows
