import numpy as np
import pandas as pd
import pytest
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from python.src.utils.functions.gpu_features import GPUFeatureEngineer

# Strategy for generating valid price series
# Prices must be positive, sequences must be long enough for lookback
price_strategy = arrays(
    dtype=np.float64,
    shape=st.integers(min_value=50, max_value=1000),
    elements=st.floats(
        min_value=0.1, max_value=10000.0, allow_nan=False, allow_infinity=False
    ),
)


@pytest.fixture
def gpu_engineer():
    # Force CPU for consistent testing environment, or detect CUDA if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return GPUFeatureEngineer(device=device)


@given(prices=price_strategy, window=st.integers(min_value=2, max_value=40))
@settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_sma_properties(gpu_engineer, prices, window):
    """
    Property: SMA should match Pandas rolling mean (within float tolerance).
    Property: output length should match input length.
    """
    if len(prices) < window:
        return  # Skip invalid windows

    # Calculate via GPU Engineer
    sma_tensor = gpu_engineer.moving_average(
        torch.tensor(prices, dtype=torch.float32), window
    )
    sma_result = sma_tensor.cpu().numpy()

    # Calculate via Pandas (Oracle)
    expected_sma = pd.Series(prices).rolling(window=window).mean().fillna(0).values

    # Check length
    assert len(sma_result) == len(prices)

    # Check values (ignoring initial NaN/zeros difference in implementation details if any)
    # GPU implementation might return 0 or first value for initial window
    # We compare validity from window-1 onwards
    np.testing.assert_allclose(
        sma_result[window - 1 :], expected_sma[window - 1 :], rtol=1e-5, atol=1e-5
    )


@given(prices=price_strategy, window=st.integers(min_value=2, max_value=40))
@settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_rsi_properties(gpu_engineer, prices, window):
    """
    Property: RSI should be between 0 and 100.
    """
    if len(prices) < window + 1:
        return

    rsi_tensor = gpu_engineer.rsi(torch.tensor(prices, dtype=torch.float32), window)
    rsi_result = rsi_tensor.cpu().numpy()

    # Invalidate initial warmup period which might remain 0 or NaN depending on implementation
    valid_rsi = rsi_result[window:]

    # Check bounds
    assert np.all(valid_rsi >= 0.0)
    assert np.all(valid_rsi <= 100.0)


@given(prices=price_strategy)
@settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_tensor_conversion_roundtrip(gpu_engineer, prices):
    """
    Property: Converting to inputs and back should preserve data.
    """
    tensor = gpu_engineer._to_tensor(prices)
    assert isinstance(tensor, torch.Tensor)
    assert len(tensor) == len(prices)

    # Check values
    np.testing.assert_allclose(tensor.cpu().numpy(), prices, rtol=1e-5)
