"""
Fixtures for environment module testing.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest
from numpy.typing import NDArray

__all__ = ["clob_env", "market_data", "mock_rust_env", "polymarket_env", "sample_prices", "trading_env", "trading_env_config"]


@pytest.fixture
def sample_prices() -> dict[str, NDArray[np.float64]]:
    """Generate various synthetic price series for testing."""
    np.random.seed(42)

    return {
        "trending_up": np.array(
            [100.0 + i * 0.5 for i in range(200)], dtype=np.float64
        ),
        "trending_down": np.array(
            [200.0 - i * 0.3 for i in range(200)], dtype=np.float64
        ),
        "mean_reverting": np.array(
            [100.0 + 10.0 * np.sin(i / 10.0) for i in range(200)], dtype=np.float64
        ),
        "volatile": np.cumsum(np.random.randn(200) * 2.0) + 100.0,
        "stable": np.array([100.0] * 200, dtype=np.float64),
        "small": np.array([100.0, 101.0, 102.0, 101.5, 103.0], dtype=np.float64),
    }


@pytest.fixture
def trading_env_config() -> dict[str, float | int]:
    """Default configuration for TradingEnv."""
    return {
        "initial_capital": 10000.0,
        "transaction_cost": 0.001,
        "lookback": 30,
        "max_steps": 100,
    }


@pytest.fixture
def trading_env(trading_env_config: dict[str, float | int], sample_prices):
    """Pre-initialized TradingEnv instance."""
    from python.src.envs.envs import TradingEnv

    env = TradingEnv(
        prices=sample_prices["trending_up"],
        initial_capital=float(trading_env_config["initial_capital"]),
        transaction_cost=float(trading_env_config["transaction_cost"]),
        lookback=int(trading_env_config["lookback"]),
        max_steps=int(trading_env_config["max_steps"]),
    )
    return env


@pytest.fixture
def clob_env(trading_env_config: dict[str, float | int], sample_prices):
    """Pre-initialized ClobEnv instance."""
    from python.src.envs.envs import ClobEnv

    env = ClobEnv(
        prices=sample_prices["volatile"],
        initial_capital=float(trading_env_config["initial_capital"]),
        transaction_cost=float(trading_env_config["transaction_cost"]),
        lookback=int(trading_env_config["lookback"]),
        max_steps=int(trading_env_config["max_steps"]),
    )
    return env


@pytest.fixture
def polymarket_env():
    """Pre-initialized PolymarketEnv instance."""
    from python.src.envs.envs import PolymarketEnv

    market_ids = ["market_1", "market_2", "market_3"]
    env = PolymarketEnv(
        market_ids=market_ids,
        initial_collateral=10000.0,
        taker_fee=0.001,
    )
    return env


@pytest.fixture
def market_data() -> dict[str, list[str] | dict[str, list[float]]]:
    """Sample Polymarket data with market IDs and price histories."""
    return {
        "market_ids": ["PRES2024", "BTC100K", "FED_RATE"],
        "prices": {
            "PRES2024": [0.45, 0.47, 0.48, 0.50, 0.52, 0.51],
            "BTC100K": [0.30, 0.32, 0.35, 0.38, 0.40, 0.42],
            "FED_RATE": [0.60, 0.58, 0.55, 0.54, 0.52, 0.50],
        },
    }


@pytest.fixture
def mock_rust_env() -> MagicMock:
    """Mock for the Rust environment exposed via PyO3."""
    env = MagicMock()
    env.step = MagicMock(return_value=(np.zeros((60,)), 0.0, False, False, {}))
    env.reset = MagicMock(return_value=np.zeros((60,)))
    env.portfolio_value = MagicMock(return_value=10000.0)
    return env
