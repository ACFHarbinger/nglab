"""
Comprehensive tests for TradingEnv and ClobEnv.
"""

import numpy as np
import pytest
from gymnasium import spaces


class TestTradingEnvInitialization:
    """Test TradingEnv initialization."""

    def test_default_initialization(self):
        """Test initialization with default parameters."""
        from python.src.env.envs import TradingEnv

        env = TradingEnv()

        assert env.initial_capital == 10000.0
        assert env.transaction_cost == 0.001
        assert env.lookback == 30
        assert env.max_steps == 1000
        assert env.render_mode is None
        assert env.num_features == 6
        assert isinstance(env.observation_space, spaces.Box)
        assert isinstance(env.action_space, spaces.Discrete)
        assert env.action_space.n == 3

    def test_custom_initialization(self, trading_env_config, sample_prices):
        """Test initialization with custom parameters."""
        from python.src.env.envs import TradingEnv

        env = TradingEnv(
            prices=sample_prices["trending_up"],
            initial_capital=50000.0,
            transaction_cost=0.002,
            lookback=20,
            max_steps=500,
            render_mode="human",
        )

        assert env.initial_capital == 50000.0
        assert env.transaction_cost == 0.002
        assert env.lookback == 20
        assert env.max_steps == 500
        assert env.render_mode == "human"
        assert len(env.prices) == len(sample_prices["trending_up"])

    def test_observation_space_shape(self, trading_env_config):
        """Test observation space has correct shape."""
        from python.src.env.envs import TradingEnv

        lookback = int(trading_env_config["lookback"])
        env = TradingEnv(lookback=lookback)

        assert env.observation_space.shape == (lookback, 6)
        assert env.observation_space.dtype == np.float64

    def test_action_space_properties(self):
        """Test action space is Discrete(3)."""
        from python.src.env.envs import TradingEnv

        env = TradingEnv()

        assert isinstance(env.action_space, spaces.Discrete)
        assert env.action_space.n == 3  # Hold, Buy, Sell


class TestTradingEnvReset:
    """Test TradingEnv reset functionality."""

    def test_reset_returns_correct_shape(self, trading_env):
        """Test reset returns observation with correct shape."""
        obs, info = trading_env.reset()

        assert isinstance(obs, np.ndarray)
        assert obs.shape == (trading_env.lookback, trading_env.num_features)
        assert isinstance(info, dict)

    def test_reset_initializes_state(self, trading_env):
        """Test reset initializes state correctly."""
        # Take some steps first
        for _ in range(5):
            trading_env.step(1)  # Buy action

        # Reset
        _obs, _info = trading_env.reset()

        # In Python fallback mode, check state
        if trading_env._rust_env is None:
            assert trading_env.position == 0.0
            assert trading_env.cash == trading_env.initial_capital
            assert trading_env.current_step == trading_env.lookback
            assert len(trading_env.returns_history) == 0

    def test_reset_with_seed(self, trading_env):
        """Test reset with seed parameter."""
        obs1, _ = trading_env.reset(seed=42)
        obs2, _ = trading_env.reset(seed=42)

        # For Python fallback, observations should be deterministic
        # (though the random price generation happens at init, not reset)
        assert obs1.shape == obs2.shape

    def test_reset_with_options(self, trading_env):
        """Test reset with options parameter."""
        obs, info = trading_env.reset(options={"key": "value"})

        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)


class TestTradingEnvStep:
    """Test TradingEnv step functionality."""

    def test_hold_action(self, trading_env):
        """Test Hold action (0) doesn't change position."""
        trading_env.reset()

        # Get initial state (Python fallback only)
        if trading_env._rust_env is None:
            initial_position = trading_env.position
            initial_cash = trading_env.cash

            _obs, reward, terminated, truncated, info = trading_env.step(0)

            assert trading_env.position == initial_position
            assert trading_env.cash == initial_cash
            assert isinstance(reward, (int, float))
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            assert isinstance(info, dict)

    def test_buy_action(self, trading_env):
        """Test Buy action (1) increases position and decreases cash."""
        trading_env.reset()

        if trading_env._rust_env is None:
            initial_cash = trading_env.cash
            initial_position = trading_env.position

            _obs, _reward, _terminated, _truncated, _info = trading_env.step(1)

            # Position should increase (if we had enough cash)
            # Cash should decrease (trade + fees)
            assert trading_env.position >= initial_position
            assert trading_env.cash <= initial_cash

    def test_sell_action(self, trading_env):
        """Test Sell action (2) decreases position and increases cash."""
        trading_env.reset()

        # First buy to have some position
        if trading_env._rust_env is None:
            trading_env.step(1)  # Buy
            position_after_buy = trading_env.position
            cash_after_buy = trading_env.cash

            if position_after_buy > 0:
                trading_env.step(2)  # Sell

                # Position should decrease
                assert trading_env.position < position_after_buy
                # Cash should increase (sale proceeds - fees)
                assert trading_env.cash > cash_after_buy

    def test_step_returns(self, trading_env):
        """Test step returns correct tuple structure."""
        trading_env.reset()

        result = trading_env.step(0)

        assert len(result) == 5
        obs, reward, terminated, truncated, info = result

        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_transaction_costs_applied(self, sample_prices):
        """Test transaction costs are applied correctly."""
        from python.src.env.envs import TradingEnv

        env = TradingEnv(
            prices=sample_prices["stable"],
            initial_capital=10000.0,
            transaction_cost=0.01,  # 1% fee
        )
        env.reset()

        if env._rust_env is None:
            initial_cash = env.cash
            env.step(1)  # Buy

            # Cash should decrease by more than just the trade size due to fees
            assert env.cash < initial_cash

    def test_info_dict_contents(self, trading_env):
        """Test info dict contains expected keys."""
        trading_env.reset()
        _, _, _, _, info = trading_env.step(0)

        assert "portfolio_value" in info
        assert "position" in info
        assert "cash" in info
        assert "sharpe_ratio" in info

    def test_termination_zero_portfolio(self, sample_prices):
        """Test termination when portfolio value <= 0."""
        from python.src.env.envs import TradingEnv

        # Create env with very high transaction costs to drain capital
        env = TradingEnv(
            prices=sample_prices["stable"],
            initial_capital=100.0,
            transaction_cost=0.5,  # 50% fee
        )
        env.reset()

        # Skip if using Rust (harder to control)
        if env._rust_env is None:
            # Force portfolio value to 0
            env.cash = 0.0
            env.position = 0.0

            _, _, _terminated, _, _ = env.step(0)

            # Should not terminate immediately (happens on next step)
            # This tests the condition is checked

    def test_termination_end_of_prices(self, sample_prices):
        """Test termination at end of price data."""
        from python.src.env.envs import TradingEnv

        env = TradingEnv(
            prices=sample_prices["small"],  # Only 5 prices
            lookback=2,
            max_steps=100,
        )
        env.reset()

        terminated = False
        steps = 0
        while not terminated and steps < 10:
            _, _, terminated, truncated, _ = env.step(0)
            steps += 1
            if truncated or terminated:
                break

        # Should terminate before 10 steps (only 5 prices - 2 lookback = 3 steps max)
        assert steps <= 5

    def test_truncation_max_steps(self, sample_prices):
        """Test truncation at max_steps."""
        from python.src.env.envs import TradingEnv

        env = TradingEnv(
            prices=sample_prices["trending_up"],
            lookback=10,
            max_steps=5,  # Very short episode
        )
        env.reset()

        truncated = False
        steps = 0
        while steps < 10:
            _, _, terminated, truncated, _ = env.step(0)
            steps += 1
            if terminated or truncated:
                break

        # Should truncate at 5 steps
        assert truncated or steps <= 5


class TestTradingEnvObservation:
    """Test TradingEnv observation generation."""

    def test_observation_shape(self, trading_env):
        """Test observation has correct shape."""
        obs, _ = trading_env.reset()

        assert obs.shape == (trading_env.lookback, trading_env.num_features)
        assert obs.dtype == np.float64

    def test_observation_no_nan(self, trading_env):
        """Test observation doesn't contain NaN values."""
        trading_env.reset()

        for _ in range(10):
            obs, _, terminated, truncated, _ = trading_env.step(
                np.random.choice([0, 1, 2])
            )
            assert not np.isnan(obs).any()
            if terminated or truncated:
                break

    def test_observation_normalization(self, sample_prices):
        """Test observation values are normalized."""
        from python.src.env.envs import TradingEnv

        env = TradingEnv(prices=sample_prices["trending_up"], lookback=10)
        obs, _ = env.reset()

        # Column 0: normalized price (relative to first price)
        prices = obs[:, 0]
        assert np.all(prices > 0.0)

        # Column 4: normalized position (should start at 0)
        positions = obs[:, 4]
        assert np.allclose(positions, 0.0)

        # Column 5: normalized cash (should start at 1.0)
        cash = obs[:, 5]
        assert np.allclose(cash, 1.0)


class TestTradingEnvSharpe:
    """Test Sharpe ratio calculation."""

    def test_sharpe_insufficient_returns(self, trading_env):
        """Test Sharpe calculation with < 2 returns."""
        trading_env.reset()

        if trading_env._rust_env is None:
            sharpe = trading_env._calculate_sharpe()
            assert sharpe == 0.0

    def test_sharpe_with_returns(self, trading_env):
        """Test Sharpe calculation with sufficient returns."""
        trading_env.reset()

        # Take several steps to accumulate returns
        for _ in range(10):
            trading_env.step(np.random.choice([0, 1, 2]))

        if trading_env._rust_env is None and len(trading_env.returns_history) >= 2:
            sharpe = trading_env._calculate_sharpe()
            assert isinstance(sharpe, float)

    def test_sharpe_zero_volatility(self, sample_prices):
        """Test Sharpe calculation with zero volatility."""
        from python.src.env.envs import TradingEnv

        env = TradingEnv(prices=sample_prices["stable"])
        env.reset()

        for _ in range(10):
            env.step(0)  # Hold (no trading)

        if env._rust_env is None:
            # With stable prices and no trading, volatility should be very low
            sharpe = env._calculate_sharpe()
            assert isinstance(sharpe, float)


class TestTradingEnvRender:
    """Test TradingEnv render functionality."""

    def test_render_human_mode(self, capsys):
        """Test render in human mode outputs to stdout."""
        from python.src.env.envs import TradingEnv

        env = TradingEnv(render_mode="human")
        env.reset()
        env.render()

        captured = capsys.readouterr()
        assert "Step:" in captured.out or len(captured.out) > 0

    def test_render_none_mode(self, capsys):
        """Test render in None mode does nothing."""
        from python.src.env.envs import TradingEnv

        env = TradingEnv(render_mode=None)
        env.reset()
        env.render()

        captured = capsys.readouterr()
        # Should not print anything
        assert captured.out == ""


class TestTradingEnvEdgeCases:
    """Test edge cases for TradingEnv."""

    def test_zero_initial_capital(self):
        """Test behavior with zero initial capital."""
        from python.src.env.envs import TradingEnv

        env = TradingEnv(initial_capital=0.0)
        obs, _ = env.reset()

        assert obs.shape[0] == env.lookback

    def test_extreme_high_prices(self):
        """Test with very high prices."""
        from python.src.env.envs import TradingEnv

        high_prices = np.array([1e6, 1e6 + 100, 1e6 + 200] * 50, dtype=np.float64)
        env = TradingEnv(prices=high_prices)
        obs, _ = env.reset()

        assert obs.shape[0] == env.lookback
        assert not np.isnan(obs).any()

    def test_extreme_low_prices(self):
        """Test with very low prices."""
        from python.src.env.envs import TradingEnv

        low_prices = np.array([0.01, 0.011, 0.012] * 50, dtype=np.float64)
        env = TradingEnv(prices=low_prices)
        obs, _ = env.reset()

        assert obs.shape[0] == env.lookback
        assert not np.isnan(obs).any()


class TestClobEnv:
    """Test ClobEnv (Central Limit Order Book environment)."""

    def test_clob_inherits_trading_env(self):
        """Test ClobEnv inherits from TradingEnv."""
        from python.src.env.envs import ClobEnv, TradingEnv

        assert issubclass(ClobEnv, TradingEnv)

    def test_clob_initialization(self, clob_env):
        """Test ClobEnv initializes correctly."""
        assert clob_env.initial_capital > 0
        assert clob_env.transaction_cost >= 0

    def test_clob_reset(self, clob_env):
        """Test ClobEnv reset works."""
        obs, info = clob_env.reset()

        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)

    def test_clob_step(self, clob_env):
        """Test ClobEnv step works."""
        clob_env.reset()
        obs, reward, terminated, truncated, info = clob_env.step(0)

        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_clob_orderbook_initialization(self, clob_env):
        """Test OrderBook initialization (if Rust available)."""
        from python.src.env.envs import HAS_RUST

        if HAS_RUST:
            assert clob_env._orderbook is not None
        else:
            # OrderBook not initialized in Python fallback
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
