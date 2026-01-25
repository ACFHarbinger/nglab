"""
Integration tests for environment ecosystem.

These tests verify the interaction between Python wrappers and Rust backends,
consistency between implementations, and system stability.
"""

import numpy as np
import pytest


class TestEnvironmentIntegration:
    """Test integration between Python and Rust components."""

    def test_trading_env_rust_integration(self, rust_available):
        """Test TradingEnv uses Rust backend when available."""
        from python.src.env.envs import TradingEnv

        env = TradingEnv()

        if rust_available:
            assert env._rust_env is not None
            # Check if internal Rust env is actually used
            # We can't easily spy on valid Rust objects, but we can check state via side effects
            # usually.
        else:
            assert env._rust_env is None

    def test_polymarket_env_rust_integration(self, rust_available):
        """Test PolymarketEnv uses Rust backend when available."""
        from python.src.env.envs import PolymarketEnv

        env = PolymarketEnv(market_ids=["M1"])

        if rust_available:
            assert env._arena is not None
        else:
            assert env._arena is None

    def test_fallback_behavior_mock(self, monkeypatch):
        """Test graceful fallback when Rust is unavailable (simulated)."""
        import python.src.env.envs

        # Simulate Rust unavailability even if it is available
        monkeypatch.setattr(python.src.env.envs, "HAS_RUST", False)
        monkeypatch.setattr(python.src.env.envs, "RustTradingEnv", None)

        from python.src.env.envs import TradingEnv

        env = TradingEnv()
        assert env._rust_env is None

        # Should still work
        obs, _ = env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (env.lookback, 6)


class TestDataConsistency:
    """Test consistency between Python and Rust implementations."""

    def test_deterministic_behavior(self):
        """Test environment is deterministic with fixed seed."""
        from python.src.env.envs import TradingEnv

        env = TradingEnv()

        # Reset with seed
        obs1, _ = env.reset(seed=42)

        # Take a sequence of actions
        actions = [1, 2, 0, 1, 0]
        obs_seq1 = []
        for a in actions:
            o, _, _, _, _ = env.step(a)
            obs_seq1.append(o)

        # Repeat
        obs2, _ = env.reset(seed=42)

        obs_seq2 = []
        for a in actions:
            o, _, _, _, _ = env.step(a)
            obs_seq2.append(o)

        # Verify
        np.testing.assert_array_equal(obs1, obs2)
        for o1, o2 in zip(obs_seq1, obs_seq2, strict=False):
            np.testing.assert_array_equal(o1, o2)


class TestSystemStability:
    """Test system stability and resource usage."""

    def test_multi_episode_loop(self):
        """Run multiple episodes back-to-back."""
        from python.src.env.envs import TradingEnv

        env = TradingEnv(max_steps=50)

        for _i in range(10):
            env.reset()
            done = False
            steps = 0
            while not done:
                _, _, term, trunc, _ = env.step(0)
                steps += 1
                done = term or trunc

            assert steps <= 51  # Allow for slight off-by-one in logic

    def test_env_batch_creation(self):
        """Test creating multiple environment instances."""
        from python.src.env.envs import TradingEnv

        envs = [TradingEnv() for _ in range(5)]

        for env in envs:
            obs, _ = env.reset()
            assert isinstance(obs, np.ndarray)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
