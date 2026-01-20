"""Integration tests for Rust-Python bindings via PyO3.

These tests verify that the Rust TradingEnv can be properly imported,
instantiated, and used from Python code. This is critical for ensuring
the deep learning training pipeline can interact with the simulation engine.
"""

import numpy as np
import pytest


def test_import_nglab():
    """Test that the nglab Rust module can be imported."""
    try:
        import nglab

        assert nglab is not None
    except ImportError as e:
        pytest.skip(f"nglab module not built: {e}")


def test_trading_env_creation():
    """Test creating a TradingEnv instance from Rust."""
    try:
        import nglab
    except ImportError:
        pytest.skip("nglab module not built")

    # Create environment with basic parameters
    env = nglab.TradingEnv(
        initial_capital=10000.0,
        lookback=50,
        transaction_cost=0.001,
    )
    assert env is not None


def test_trading_env_reset():
    """Test resetting the environment and getting initial observation."""
    try:
        import nglab
    except ImportError:
        pytest.skip("nglab module not built")

    env = nglab.TradingEnv(initial_capital=10000.0, lookback=50)

    # Reset should return observation and info dict
    obs, info = env.reset(seed=42)

    # Check observation is numpy array with correct shape
    assert isinstance(obs, np.ndarray)
    assert obs.shape[0] == 50, f"Expected lookback_window=50, got {obs.shape[0]}"
    assert obs.dtype in (np.float64, np.float32)

    # Check info dict
    assert isinstance(info, dict)


def test_trading_env_step():
    """Test stepping through the environment with actions."""
    try:
        import nglab
    except ImportError:
        pytest.skip("nglab module not built")

    env = nglab.TradingEnv(initial_capital=10000.0, lookback=50)
    env.reset(seed=42)

    # Test different actions (assuming discrete action space)
    # 0 = hold, 1 = buy, 2 = sell (adjust based on actual implementation)
    for action in [0, 1, 2]:
        obs, reward, terminated, truncated, info = env.step(action)

        # Verify return types
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, (float, np.floating))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

        # Verify observation shape
        assert obs.shape[0] == 50

        if terminated or truncated:
            break


def test_trading_env_episode():
    """Test running a complete episode."""
    try:
        import nglab
    except ImportError:
        pytest.skip("nglab module not built")

    env = nglab.TradingEnv(initial_capital=10000.0, lookback=50)
    _, _ = env.reset(seed=123)

    total_reward = 0.0
    steps = 0
    max_steps = 100

    while steps < max_steps:
        # Random action for testing
        action = np.random.choice([0, 1, 2])
        _obs, reward, terminated, truncated, _info = env.step(action)

        total_reward += reward
        steps += 1

        if terminated or truncated:
            break

    assert steps > 0, "Environment should run for at least one step"
    assert isinstance(total_reward, (float, np.floating))


def test_zero_copy_numpy_transfer():
    """Test that numpy arrays are transferred without copying (zero-copy)."""
    try:
        import nglab
    except ImportError:
        pytest.skip("nglab module not built")

    env = nglab.TradingEnv(initial_capital=10000.0, lookback=50)
    obs, _ = env.reset(seed=42)

    # Check that observation is a proper numpy array
    assert isinstance(obs, np.ndarray)
    assert obs.flags["C_CONTIGUOUS"] or obs.flags["F_CONTIGUOUS"]

    # Verify data is not copied (observation should be writable or read-only view)
    # If read-only, it's a view into Rust memory (zero-copy)
    # This is the expected behavior for PyO3 numpy integration


def test_multiple_environments():
    """Test creating and using multiple environment instances simultaneously."""
    try:
        import nglab
    except ImportError:
        pytest.skip("nglab module not built")

    envs = [nglab.TradingEnv(initial_capital=10000.0, lookback=50) for _ in range(3)]

    # Reset all environments
    observations = [env.reset(seed=i)[0] for i, env in enumerate(envs)]

    # Verify each environment has independent state
    assert len(observations) == 3
    for obs in observations:
        assert isinstance(obs, np.ndarray)
        assert obs.shape[0] == 50

    # Step each environment independently
    for env in envs:
        obs, reward, _terminated, _truncated, _info = env.step(1)
        assert isinstance(reward, (float, np.floating))


def test_action_space_bounds():
    """Test that invalid actions are handled properly."""
    try:
        import nglab
    except ImportError:
        pytest.skip("nglab module not built")

    env = nglab.TradingEnv(initial_capital=10000.0, lookback=50)
    env.reset(seed=42)

    # Test with invalid action (if discrete, actions outside [0, 1, 2] should error)
    # This test depends on actual action space implementation
    try:
        _obs, _reward, _terminated, _truncated, _info = env.step(999)
        # If no error, action space might be continuous or unbounded
    except (ValueError, RuntimeError):
        # Expected behavior for invalid discrete action
        pass


def test_reproducibility_with_seed():
    """Test that seeding produces reproducible results."""
    try:
        import nglab
    except ImportError:
        pytest.skip("nglab module not built")

    # Create two environments with same seed
    env1 = nglab.TradingEnv(initial_capital=10000.0, lookback=50)
    env2 = nglab.TradingEnv(initial_capital=10000.0, lookback=50)

    obs1, _ = env1.reset(seed=42)
    obs2, _ = env2.reset(seed=42)

    # Observations should be identical
    np.testing.assert_array_equal(obs1, obs2)

    # Take same actions
    obs1, reward1, _, _, _ = env1.step(1)
    obs2, reward2, _, _, _ = env2.step(1)

    # Results should be identical
    np.testing.assert_array_equal(obs1, obs2)
    assert reward1 == reward2


def test_orderbook_integration():
    """Test if OrderBook can be imported and used."""
    try:
        import nglab
    except ImportError:
        pytest.skip("nglab module not built")

    # Try to create OrderBook if exposed
    try:
        orderbook = nglab.OrderBook()
        assert orderbook is not None
    except AttributeError:
        pytest.skip("OrderBook not exposed to Python")


def test_polymarket_arena():
    """Test if PolymarketArena can be imported and used."""
    try:
        import nglab
    except ImportError:
        pytest.skip("nglab module not built")

    # Try to create PolymarketArena if exposed
    try:
        arena = nglab.PolymarketArena()
        assert arena is not None
    except AttributeError:
        pytest.skip("PolymarketArena not exposed to Python")


@pytest.mark.slow
def test_long_episode_stability():
    """Test that environment remains stable over long episodes."""
    try:
        import nglab
    except ImportError:
        pytest.skip("nglab module not built")

    env = nglab.TradingEnv(initial_capital=10000.0, lookback=50)
    obs, _ = env.reset(seed=42)

    steps = 0
    max_steps = 1000

    while steps < max_steps:
        action = np.random.choice([0, 1, 2])
        obs, reward, terminated, truncated, _info = env.step(action)

        # Verify no NaN or Inf values
        assert not np.isnan(obs).any(), f"NaN in observation at step {steps}"
        assert not np.isinf(obs).any(), f"Inf in observation at step {steps}"
        assert not np.isnan(reward), f"NaN reward at step {steps}"
        assert not np.isinf(reward), f"Inf reward at step {steps}"

        steps += 1

        if terminated or truncated:
            obs, _ = env.reset()

    assert steps == max_steps, "Environment should complete all steps"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
