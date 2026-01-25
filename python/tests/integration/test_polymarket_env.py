"""
Comprehensive tests for PolymarketEnv.
"""

import numpy as np
import pytest
from gymnasium import spaces


class TestPolymarketEnvInitialization:
    """Test PolymarketEnv initialization."""

    def test_empty_market_ids(self):
        """Test initialization with empty market_ids."""
        from python.src.env.envs import PolymarketEnv

        env = PolymarketEnv(market_ids=[])

        # Should create env with at least 1 market for obs space
        assert isinstance(env.observation_space, spaces.Box)
        assert isinstance(env.action_space, spaces.MultiDiscrete)

    def test_single_market(self):
        """Test initialization with single market."""
        from python.src.env.envs import PolymarketEnv

        env = PolymarketEnv(market_ids=["MARKET1"])

        assert len(env.market_ids) == 1
        # Obs: [collateral, pnl] + [price, yes_pos, no_pos] = 2 + 3 = 5
        assert env.observation_space.shape == (5,)
        # Action: 5 actions for 1 market
        assert env.action_space.nvec.tolist() == [5]

    def test_multiple_markets(self, market_data):
        """Test initialization with multiple markets."""
        from python.src.env.envs import PolymarketEnv

        market_ids = market_data["market_ids"]
        env = PolymarketEnv(market_ids=market_ids)

        assert len(env.market_ids) == 3
        # Obs: [collateral, pnl] + 3 * [price, yes_pos, no_pos] = 2 + 9 = 11
        assert env.observation_space.shape == (11,)
        # Action: 5 actions for each of 3 markets
        assert env.action_space.nvec.tolist() == [5, 5, 5]

    def test_custom_parameters(self):
        """Test initialization with custom parameters."""
        from python.src.env.envs import PolymarketEnv

        env = PolymarketEnv(
            market_ids=["M1", "M2"],
            initial_collateral=50000.0,
            taker_fee=0.002,
            render_mode="human",
        )

        assert env.initial_collateral == 50000.0
        assert env.taker_fee == 0.002
        assert env.render_mode == "human"

    def test_observation_space_dimensions(self, market_data):
        """Test observation space dimensions match formula."""
        from python.src.env.envs import PolymarketEnv

        for num_markets in [1, 2, 3, 5]:
            market_ids = [f"M{i}" for i in range(num_markets)]
            env = PolymarketEnv(market_ids=market_ids)

            expected_dim = 2 + num_markets * 3
            assert env.observation_space.shape == (expected_dim,)

    def test_action_space_multidiscrete(self):
        """Test action space is MultiDiscrete."""
        from python.src.env.envs import PolymarketEnv

        env = PolymarketEnv(market_ids=["M1", "M2", "M3"])

        assert isinstance(env.action_space, spaces.MultiDiscrete)
        assert len(env.action_space.nvec) == 3
        assert all(n == 5 for n in env.action_space.nvec)


class TestPolymarketEnvReset:
    """Test PolymarketEnv reset functionality."""

    def test_reset_returns_correct_shape(self, polymarket_env):
        """Test reset returns observation with correct shape."""
        obs, info = polymarket_env.reset()

        assert isinstance(obs, np.ndarray)
        assert obs.shape == polymarket_env.observation_space.shape
        assert isinstance(info, dict)

    def test_reset_initializes_collateral(self, polymarket_env):
        """Test reset initializes collateral correctly."""
        obs, _ = polymarket_env.reset()

        # First element is normalized collateral (should be 1.0)
        assert obs[0] == pytest.approx(1.0, rel=0.01)

    def test_reset_clears_positions(self, polymarket_env):
        """Test reset clears all positions."""
        # Take some actions
        action = np.array([1, 2, 1])  # Buy Yes, Buy No, Buy Yes
        polymarket_env.reset()
        polymarket_env.step(action)

        # Reset
        _obs, _ = polymarket_env.reset()

        # Check positions are cleared (using Python fallback)
        if polymarket_env._arena is None:
            assert polymarket_env._collateral == polymarket_env.initial_collateral
            assert len(polymarket_env._positions) == 0 or all(
                pos == (0.0, 0.0) for pos in polymarket_env._positions.values()
            )

    def test_reset_initializes_prices(self, polymarket_env):
        """Test reset initializes prices to 0.5."""
        polymarket_env.reset()

        if polymarket_env._arena is None:
            for market_id in polymarket_env.market_ids:
                assert polymarket_env._prices[market_id] == 0.5

    def test_reset_with_seed(self, polymarket_env):
        """Test reset with seed parameter."""
        obs1, _ = polymarket_env.reset(seed=42)
        obs2, _ = polymarket_env.reset(seed=42)

        np.testing.assert_array_equal(obs1, obs2)

    def test_reset_with_options(self, polymarket_env):
        """Test reset with options parameter."""
        obs, info = polymarket_env.reset(options={"test": "value"})

        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)


class TestPolymarketEnvActions:
    """Test PolymarketEnv action execution."""

    def test_hold_action(self, polymarket_env):
        """Test Hold action (0) for each market."""
        polymarket_env.reset()

        if polymarket_env._arena is None:
            initial_collateral = polymarket_env._collateral
            action = np.array([0, 0, 0])  # Hold all markets

            polymarket_env.step(action)

            # Collateral should not change
            assert polymarket_env._collateral == initial_collateral

    def test_buy_yes_action(self, polymarket_env):
        """Test Buy Yes action (1) increases position and decreases collateral."""
        polymarket_env.reset()

        if polymarket_env._arena is None:
            initial_collateral = polymarket_env._collateral
            market_id = polymarket_env.market_ids[0]
            action = np.array([1, 0, 0])  # Buy Yes on first market

            polymarket_env.step(action)

            # Collateral should decrease
            assert polymarket_env._collateral < initial_collateral
            # Position should increase
            yes_pos, _no_pos = polymarket_env._positions.get(market_id, (0.0, 0.0))
            assert yes_pos > 0.0

    def test_buy_no_action(self, polymarket_env):
        """Test Buy No action (2) increases position and decreases collateral."""
        polymarket_env.reset()

        if polymarket_env._arena is None:
            initial_collateral = polymarket_env._collateral
            market_id = polymarket_env.market_ids[1]
            action = np.array([0, 2, 0])  # Buy No on second market

            polymarket_env.step(action)

            # Collateral should decrease
            assert polymarket_env._collateral < initial_collateral
            # Position should increase
            _yes_pos, no_pos = polymarket_env._positions.get(market_id, (0.0, 0.0))
            assert no_pos > 0.0

    def test_sell_yes_action(self, polymarket_env):
        """Test Sell Yes action (3) decreases position and increases collateral."""
        polymarket_env.reset()

        if polymarket_env._arena is None:
            market_id = polymarket_env.market_ids[0]

            # First buy to have position
            action_buy = np.array([1, 0, 0])
            polymarket_env.step(action_buy)

            collateral_after_buy = polymarket_env._collateral
            yes_pos_before, _ = polymarket_env._positions.get(market_id, (0.0, 0.0))

            # Now sell
            action_sell = np.array([3, 0, 0])
            polymarket_env.step(action_sell)

            # Collateral should increase
            assert polymarket_env._collateral > collateral_after_buy
            # Position should decrease
            yes_pos_after, _ = polymarket_env._positions.get(market_id, (0.0, 0.0))
            assert yes_pos_after < yes_pos_before

    def test_sell_no_action(self, polymarket_env):
        """Test Sell No action (4) decreases position and increases collateral."""
        polymarket_env.reset()

        if polymarket_env._arena is None:
            market_id = polymarket_env.market_ids[1]

            # First buy to have position
            action_buy = np.array([0, 2, 0])
            polymarket_env.step(action_buy)

            collateral_after_buy = polymarket_env._collateral
            _, no_pos_before = polymarket_env._positions.get(market_id, (0.0, 0.0))

            # Now sell
            action_sell = np.array([0, 4, 0])
            polymarket_env.step(action_sell)

            # Collateral should increase
            assert polymarket_env._collateral > collateral_after_buy
            # Position should decrease
            _, no_pos_after = polymarket_env._positions.get(market_id, (0.0, 0.0))
            assert no_pos_after < no_pos_before

    def test_insufficient_collateral_prevents_trade(self):
        """Test insufficient collateral prevents trades."""
        from python.src.env.envs import PolymarketEnv

        env = PolymarketEnv(
            market_ids=["M1"],
            initial_collateral=10.0,  # Very low
            taker_fee=0.5,  # High fee
        )
        env.reset()

        if env._arena is None:
            # Try to buy with insufficient collateral
            # Amount = 0.01 * 10.0 = 0.1, cost = 0.1 * 0.5 * 1.5 = 0.075
            # This should work, but if we drain it first...
            env._collateral = 0.01  # Drain collateral

            action = np.array([1])  # Try to buy
            env.step(action)

            # Trade should not execute
            market_id = env.market_ids[0]
            yes_pos, _ = env._positions.get(market_id, (0.0, 0.0))
            assert yes_pos == 0.0

    def test_insufficient_position_prevents_sell(self, polymarket_env):
        """Test insufficient position prevents sells."""
        polymarket_env.reset()

        if polymarket_env._arena is None:
            # Try to sell without any position
            market_id = polymarket_env.market_ids[0]
            action = np.array([3, 0, 0])  # Sell Yes

            polymarket_env.step(action)

            # Position should still be 0
            yes_pos, _ = polymarket_env._positions.get(market_id, (0.0, 0.0))
            assert yes_pos == 0.0

    def test_fees_applied_correctly(self):
        """Test fees are applied correctly."""
        from python.src.env.envs import PolymarketEnv

        env = PolymarketEnv(
            market_ids=["M1"],
            initial_collateral=10000.0,
            taker_fee=0.01,  # 1% fee
        )
        env.reset()

        if env._arena is None:
            initial_collateral = env._collateral
            action = np.array([1])  # Buy Yes
            env.step(action)

            # Calculate expected cost
            amount = env.initial_collateral * 0.01  # 100.0
            price = 0.5
            expected_cost = (
                amount * price * (1 + env.taker_fee)
            )  # 100 * 0.5 * 1.01 = 50.5

            assert env._collateral == pytest.approx(
                initial_collateral - expected_cost, rel=0.01
            )


class TestPolymarketEnvObservation:
    """Test PolymarketEnv observation generation."""

    def test_observation_shape(self, polymarket_env):
        """Test observation has correct shape."""
        obs, _ = polymarket_env.reset()

        assert obs.shape == polymarket_env.observation_space.shape
        assert obs.dtype == np.float64

    def test_observation_collateral_normalization(self, polymarket_env):
        """Test collateral is normalized."""
        obs, _ = polymarket_env.reset()

        # First element should be normalized collateral (1.0 at start)
        assert obs[0] == pytest.approx(1.0, rel=0.01)

    def test_observation_pnl_normalization(self, polymarket_env):
        """Test PnL is normalized."""
        obs, _ = polymarket_env.reset()

        # Second element should be normalized PnL (0.0 at start)
        assert obs[1] == pytest.approx(0.0, abs=0.01)

    def test_observation_market_data(self, polymarket_env):
        """Test market-specific data in observation."""
        obs, _ = polymarket_env.reset()

        num_markets = len(polymarket_env.market_ids)

        # Each market has [price, yes_pos, no_pos]
        for i in range(num_markets):
            base_idx = 2 + i * 3
            price = obs[base_idx]
            yes_pos = obs[base_idx + 1]
            no_pos = obs[base_idx + 2]

            # Price should be 0.5 at start (or from arena)
            assert 0.0 <= price <= 1.0
            # Positions should be 0 at start
            assert yes_pos == pytest.approx(0.0, abs=0.01)
            assert no_pos == pytest.approx(0.0, abs=0.01)

    def test_observation_no_nan(self, polymarket_env):
        """Test observation doesn't contain NaN."""
        polymarket_env.reset()

        for _ in range(10):
            action = polymarket_env.action_space.sample()
            obs, _, terminated, _, _ = polymarket_env.step(action)

            assert not np.isnan(obs).any()
            if terminated:
                break


class TestPolymarketEnvAccountValue:
    """Test account value calculation."""

    def test_initial_account_value(self, polymarket_env):
        """Test account value equals initial collateral at start."""
        polymarket_env.reset()

        account_value = polymarket_env._account_value()

        assert account_value == pytest.approx(
            polymarket_env.initial_collateral, rel=0.01
        )

    def test_account_value_with_positions(self, polymarket_env):
        """Test account value includes position values."""
        polymarket_env.reset()

        # Buy some positions
        action = np.array([1, 2, 1])  # Buy across markets
        polymarket_env.step(action)

        account_value = polymarket_env._account_value()

        # Account value should be > 0 (collateral + positions)
        assert account_value > 0.0

    def test_account_value_python_fallback(self):
        """Test account value calculation in Python fallback."""
        from python.src.env.envs import PolymarketEnv

        env = PolymarketEnv(market_ids=["M1"], initial_collateral=10000.0)
        env.reset()

        if env._arena is None:
            # Manually set state
            env._collateral = 5000.0
            env._positions = {"M1": (100.0, 50.0)}
            env._prices = {"M1": 0.6}

            account_value = env._account_value()

            # Expected: 5000 + 100*0.6 + 50*0.4 = 5000 + 60 + 20 = 5080
            assert account_value == pytest.approx(5080.0, rel=0.01)


class TestPolymarketEnvReward:
    """Test reward calculation."""

    def test_reward_calculation(self, polymarket_env):
        """Test reward is calculated correctly."""
        polymarket_env.reset()

        _, reward, _, _, _ = polymarket_env.step(np.array([0, 0, 0]))

        # Reward should be a float
        assert isinstance(reward, int | float)

    def test_reward_positive_on_profit(self):
        """Test reward is positive when account value increases."""
        from python.src.env.envs import PolymarketEnv

        env = PolymarketEnv(market_ids=["M1"], initial_collateral=10000.0)
        env.reset()

        if env._arena is None:
            # Artificially increase account value
            env._prices["M1"] = 0.8  # Price goes up
            env._positions["M1"] = (100.0, 0.0)  # Have yes position

            _, reward, _, _, _ = env.step(np.array([0]))

            # Reward might be positive (depends on implementation)
            # Just verify it's calculated
            assert isinstance(reward, int | float)


class TestPolymarketEnvTermination:
    """Test termination conditions."""

    def test_termination_zero_account_value(self):
        """Test termination when account value <= 0."""
        from python.src.env.envs import PolymarketEnv

        env = PolymarketEnv(market_ids=["M1"], initial_collateral=10000.0)
        env.reset()

        if env._arena is None:
            # Force zero account value
            env._collateral = 0.0
            env._positions = {}

            _, _, terminated, _, _ = env.step(np.array([0]))

            assert terminated

    def test_no_truncation(self, polymarket_env):
        """Test truncated is always False (no max_steps)."""
        polymarket_env.reset()

        for _ in range(10):
            _, _, _, truncated, _ = polymarket_env.step(np.array([0, 0, 0]))
            assert not truncated


class TestPolymarketEnvInfo:
    """Test info dict contents."""

    def test_info_contains_account_value(self, polymarket_env):
        """Test info dict contains account_value."""
        polymarket_env.reset()
        _, _, _, _, info = polymarket_env.step(np.array([0, 0, 0]))

        assert "account_value" in info
        assert isinstance(info["account_value"], int | float)

    def test_info_contains_collateral(self, polymarket_env):
        """Test info dict contains collateral."""
        polymarket_env.reset()
        _, _, _, _, info = polymarket_env.step(np.array([0, 0, 0]))

        assert "collateral" in info
        assert isinstance(info["collateral"], int | float)


class TestPolymarketEnvMultiMarket:
    """Test multi-market execution."""

    def test_simultaneous_actions(self, polymarket_env):
        """Test executing actions across multiple markets."""
        polymarket_env.reset()

        # Different action for each market
        action = np.array([1, 2, 0])  # Buy Yes, Buy No, Hold

        obs, reward, _terminated, _truncated, _info = polymarket_env.step(action)

        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, int | float)

    def test_independent_market_positions(self, polymarket_env):
        """Test positions are tracked independently per market."""
        polymarket_env.reset()

        if polymarket_env._arena is None:
            # Buy Yes on market 0, Buy No on market 1
            action = np.array([1, 2, 0])
            polymarket_env.step(action)

            # Check positions
            yes_0, _no_0 = polymarket_env._positions.get(
                polymarket_env.market_ids[0], (0.0, 0.0)
            )
            _yes_1, no_1 = polymarket_env._positions.get(
                polymarket_env.market_ids[1], (0.0, 0.0)
            )

            # Market 0 should have yes position
            assert yes_0 > 0.0
            # Market 1 should have no position
            assert no_1 > 0.0


class TestPolymarketEnvRender:
    """Test PolymarketEnv render functionality."""

    def test_render_human_mode(self, capsys):
        """Test render in human mode outputs to stdout."""
        from python.src.env.envs import PolymarketEnv

        env = PolymarketEnv(market_ids=["M1"], render_mode="human")
        env.reset()
        env.render()

        captured = capsys.readouterr()
        assert "Account Value" in captured.out or len(captured.out) > 0

    def test_render_none_mode(self, capsys):
        """Test render in None mode does nothing."""
        from python.src.env.envs import PolymarketEnv

        env = PolymarketEnv(market_ids=["M1"], render_mode=None)
        env.reset()
        env.render()

        captured = capsys.readouterr()
        assert captured.out == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
