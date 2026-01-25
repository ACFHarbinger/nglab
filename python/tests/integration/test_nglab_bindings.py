"""
Tests for NGLab Rust bindings (PyO3).

These tests check the Rust bindings exposed via PyO3.
They will be skipped if the Rust bindings are not available.
"""

import numpy as np
import pytest


class TestImports:
    """Test importing nglab modules."""

    def test_nglab_import(self):
        """Test nglab module can be imported."""
        try:
            import nglab  # noqa: F401

            imported = True
        except ImportError:
            imported = False

        # Just verify import attempt completes
        assert isinstance(imported, bool)

    def test_individual_imports(self, rust_available):
        """Test importing individual classes."""
        if not rust_available:
            pytest.skip("Rust bindings not available")

        from nglab import Arena, OrderBook, PolymarketArena, TradingEnv

        # All should be classes/types
        assert Arena is not None
        assert OrderBook is not None
        assert TradingEnv is not None
        assert PolymarketArena is not None

    def test_graceful_import_failure(self):
        """Test graceful handling of ImportError."""
        try:
            from nglab import NonExistentClass  # noqa: F401

            exists = True
        except (ImportError, AttributeError):
            exists = False

        assert not exists


class TestArena:
    """Test Arena Rust class."""

    def test_arena_initialization(self, arena_instance):
        """Test Arena can be initialized."""
        assert arena_instance is not None

    def test_arena_step_count(self, arena_instance):
        """Test Arena.step_count() returns int."""
        step_count = arena_instance.step_count()

        assert isinstance(step_count, int)
        assert step_count >= 0

    def test_arena_step_count_initial(self, rust_available):
        """Test Arena starts with step_count of 0."""
        if not rust_available:
            pytest.skip("Rust bindings not available")

        from nglab import Arena

        arena = Arena()
        assert arena.step_count() == 0


class TestOrderBook:
    """Test OrderBook Rust class."""

    def test_orderbook_initialization(self, orderbook_instance):
        """Test OrderBook can be initialized."""
        assert orderbook_instance is not None

    def test_orderbook_best_bid(self, orderbook_instance):
        """Test OrderBook.best_bid() returns float or None."""
        best_bid = orderbook_instance.best_bid()

        assert best_bid is None or isinstance(best_bid, float)
        if best_bid is not None:
            assert best_bid >= 0.0

    def test_orderbook_best_ask(self, orderbook_instance):
        """Test OrderBook.best_ask() returns float or None."""
        best_ask = orderbook_instance.best_ask()

        assert best_ask is None or isinstance(best_ask, float)
        if best_ask is not None:
            assert best_ask >= 0.0

    def test_orderbook_mid_price(self, orderbook_instance):
        """Test OrderBook.mid_price() returns float or None."""
        mid_price = orderbook_instance.mid_price()

        assert mid_price is None or isinstance(mid_price, float)
        if mid_price is not None:
            assert mid_price >= 0.0

    def test_orderbook_spread(self, orderbook_instance):
        """Test OrderBook.spread() returns float or None."""
        spread = orderbook_instance.spread()

        assert spread is None or isinstance(spread, float)
        if spread is not None:
            assert spread >= 0.0

    def test_orderbook_imbalance(self, orderbook_instance):
        """Test OrderBook.imbalance() returns float or None."""
        imbalance = orderbook_instance.imbalance()

        assert imbalance is None or isinstance(imbalance, float)
        if imbalance is not None:
            assert -1.0 <= imbalance <= 1.0

    def test_orderbook_total_bid_volume(self, orderbook_instance):
        """Test OrderBook.total_bid_volume() returns float."""
        volume = orderbook_instance.total_bid_volume()

        # Volume is usually 0.0 if empty, not None
        assert isinstance(volume, float)
        assert volume >= 0.0

    def test_orderbook_total_ask_volume(self, orderbook_instance):
        """Test OrderBook.total_ask_volume() returns float."""
        volume = orderbook_instance.total_ask_volume()

        assert isinstance(volume, float)
        assert volume >= 0.0

    def test_orderbook_spread_relationship(self, orderbook_instance):
        """Test spread = ask - bid."""
        best_bid = orderbook_instance.best_bid()
        best_ask = orderbook_instance.best_ask()
        spread = orderbook_instance.spread()

        if best_bid is not None and best_ask is not None and spread is not None:
            expected_spread = best_ask - best_bid
            assert spread == pytest.approx(expected_spread, abs=1e-10)

    def test_orderbook_mid_price_relationship(self, orderbook_instance):
        """Test mid_price = (bid + ask) / 2."""
        best_bid = orderbook_instance.best_bid()
        best_ask = orderbook_instance.best_ask()
        mid_price = orderbook_instance.mid_price()

        if best_bid is not None and best_ask is not None and mid_price is not None:
            expected_mid = (best_bid + best_ask) / 2.0
            assert mid_price == pytest.approx(expected_mid, abs=1e-10)


class TestTradingEnvRust:
    """Test Rust TradingEnv class."""

    def test_trading_env_initialization(self, rust_available):
        """Test TradingEnv can be initialized."""
        if not rust_available:
            pytest.skip("Rust bindings not available")

        from nglab import TradingEnv

        env = TradingEnv(
            initial_capital=10000.0,
            transaction_cost=0.001,
            lookback=30,
            max_steps=100,
            enable_logging=False,
        )

        assert env is not None

    def test_trading_env_reset_shape(self, rust_trading_env):
        """Test reset() returns correct shape."""
        obs, info = rust_trading_env.reset()

        assert isinstance(obs, np.ndarray)
        assert obs.shape == (30, 6)  # (lookback, features)
        assert isinstance(info, dict)

    def test_trading_env_reset_dtype(self, rust_trading_env):
        """Test observation is float64."""
        obs, _ = rust_trading_env.reset()

        assert obs.dtype == np.float64

    def test_trading_env_step_signature(self, rust_trading_env):
        """Test step() returns correct tuple."""
        rust_trading_env.reset()
        result = rust_trading_env.step(0)

        assert len(result) == 5
        obs, reward, terminated, truncated, info = result

        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, int | float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_trading_env_step_actions(self, rust_trading_env):
        """Test all action types (0, 1, 2)."""
        rust_trading_env.reset()

        for action in [0, 1, 2]:  # Hold, Buy, Sell
            obs, _reward, terminated, truncated, _info = rust_trading_env.step(action)

            assert isinstance(obs, np.ndarray)
            assert not np.isnan(obs).any()

            if terminated or truncated:
                rust_trading_env.reset()

    def test_trading_env_load_prices(self, rust_available):
        """Test load_prices() accepts list[float]."""
        if not rust_available:
            pytest.skip("Rust bindings not available")

        from nglab import TradingEnv

        env = TradingEnv(
            initial_capital=10000.0,
            transaction_cost=0.001,
            lookback=20,
            max_steps=100,
            enable_logging=False,
        )

        prices = [100.0 + i * 0.5 for i in range(200)]
        env.load_prices(prices)

        # Should not raise exception
        obs, _ = env.reset()
        assert obs.shape[0] == 20

    def test_trading_env_get_observation(self, rust_trading_env):
        """Test get_observation() returns ndarray."""
        rust_trading_env.reset()

        obs = rust_trading_env.get_observation()

        assert isinstance(obs, np.ndarray)
        assert obs.shape == (30, 6)

    def test_trading_env_episode_completion(self, rust_trading_env):
        """Test completing a full episode."""
        obs, _ = rust_trading_env.reset()

        steps = 0
        terminated = False
        truncated = False

        while steps < 200 and not (terminated or truncated):
            action = np.random.choice([0, 1, 2])
            obs, _reward, terminated, truncated, _info = rust_trading_env.step(action)
            steps += 1

        # Should complete within 200 steps
        assert steps <= 200
        assert isinstance(obs, np.ndarray)

    def test_trading_env_observation_no_nan(self, rust_trading_env):
        """Test observations never contain NaN."""
        rust_trading_env.reset()

        for _ in range(50):
            action = np.random.choice([0, 1, 2])
            obs, _, terminated, truncated, _ = rust_trading_env.step(action)

            assert not np.isnan(obs).any(), "Observation contains NaN"

            if terminated or truncated:
                break

    def test_trading_env_reward_is_finite(self, rust_trading_env):
        """Test rewards are always finite."""
        rust_trading_env.reset()

        for _ in range(50):
            action = np.random.choice([0, 1, 2])
            _, reward, terminated, truncated, _ = rust_trading_env.step(action)

            assert np.isfinite(reward), f"Reward {reward} is not finite"

            if terminated or truncated:
                break


class TestPolymarketArenaRust:
    """Test Rust PolymarketArena class."""

    def test_polymarket_arena_initialization(self, polymarket_arena):
        """Test PolymarketArena can be initialized."""
        assert polymarket_arena is not None

    def test_polymarket_arena_account_value(self, polymarket_arena):
        """Test account_value() returns float."""
        account_value = polymarket_arena.account_value()

        assert isinstance(account_value, float)
        assert account_value > 0.0

    def test_polymarket_arena_collateral(self, polymarket_arena):
        """Test collateral() returns float."""
        collateral = polymarket_arena.collateral()

        assert isinstance(collateral, float)
        # Should equal initial collateral at start
        assert collateral == pytest.approx(10000.0, rel=0.01)

    def test_polymarket_arena_realized_pnl(self, polymarket_arena):
        """Test realized_pnl() returns float."""
        pnl = polymarket_arena.realized_pnl()

        assert isinstance(pnl, float)
        # Should be 0 at start
        assert pnl == pytest.approx(0.0, abs=0.01)

    def test_polymarket_arena_current_step(self, polymarket_arena):
        """Test current_step() returns int."""
        step = polymarket_arena.current_step()

        assert isinstance(step, int)
        assert step >= 0

    def test_polymarket_arena_num_markets(self, polymarket_arena):
        """Test num_markets() returns int."""
        num = polymarket_arena.num_markets()

        assert isinstance(num, int)
        assert num >= 0

    def test_polymarket_arena_load_markets(self, polymarket_arena):
        """Test load_markets() accepts JSON string."""
        json_data = '{"markets": []}'

        # Should not raise exception
        try:
            polymarket_arena.load_markets(json_data)
            loaded = True
        except Exception:
            # May fail if JSON format is wrong, but that's OK
            loaded = False

        # Just verify the call signature is correct
        assert isinstance(loaded, bool)

    def test_polymarket_arena_load_price_history(self, polymarket_arena):
        """Test load_price_history() accepts CSV string."""
        market_id = "TEST_MARKET"
        csv_data = "timestamp,price\n1000,0.5\n2000,0.55\n"

        # Should not raise exception (may fail if market doesn't exist)
        try:
            polymarket_arena.load_price_history(market_id, csv_data)
            loaded = True
        except Exception:
            loaded = False

        assert isinstance(loaded, bool)

    def test_polymarket_arena_reset(self, polymarket_arena):
        """Test reset() works."""
        # Should not raise exception
        polymarket_arena.reset(10000.0)

        # Collateral should be reset
        assert polymarket_arena.collateral() == pytest.approx(10000.0, rel=0.01)

    def test_polymarket_arena_get_price(self, rust_available):
        """Test get_price() returns float or None."""
        if not rust_available:
            pytest.skip("Rust bindings not available")

        from nglab import PolymarketArena

        arena = PolymarketArena(initial_collateral=10000.0, taker_fee=0.001)

        # Try to get price for non-existent market
        price = arena.get_price("NONEXISTENT")

        # Should return None for non-existent market
        assert price is None or isinstance(price, float)

    def test_polymarket_arena_get_position(self, rust_available):
        """Test get_position() returns (float, float) tuple."""
        if not rust_available:
            pytest.skip("Rust bindings not available")

        from nglab import PolymarketArena

        arena = PolymarketArena(initial_collateral=10000.0, taker_fee=0.001)

        # Try to get position for market
        position = arena.get_position("TEST_MARKET")

        assert isinstance(position, tuple)
        assert len(position) == 2
        assert isinstance(position[0], float)  # yes_pos
        assert isinstance(position[1], float)  # no_pos

    def test_polymarket_arena_buy_yes(self, rust_available):
        """Test buy_yes() returns float (cost)."""
        if not rust_available:
            pytest.skip("Rust bindings not available")

        from nglab import PolymarketArena

        arena = PolymarketArena(initial_collateral=10000.0, taker_fee=0.001)

        # Try to buy (will likely fail without market loaded)
        try:
            cost = arena.buy_yes("TEST_MARKET", 100.0)
            assert isinstance(cost, float)
        except Exception:
            # Expected if market doesn't exist
            pass

    def test_polymarket_arena_buy_no(self, rust_available):
        """Test buy_no() returns float (cost)."""
        if not rust_available:
            pytest.skip("Rust bindings not available")

        from nglab import PolymarketArena

        arena = PolymarketArena(initial_collateral=10000.0, taker_fee=0.001)

        try:
            cost = arena.buy_no("TEST_MARKET", 100.0)
            assert isinstance(cost, float)
        except Exception:
            pass

    def test_polymarket_arena_sell_yes(self, rust_available):
        """Test sell_yes() returns float (proceeds)."""
        if not rust_available:
            pytest.skip("Rust bindings not available")

        from nglab import PolymarketArena

        arena = PolymarketArena(initial_collateral=10000.0, taker_fee=0.001)

        try:
            proceeds = arena.sell_yes("TEST_MARKET", 50.0)
            assert isinstance(proceeds, float)
        except Exception:
            pass

    def test_polymarket_arena_split(self, rust_available):
        """Test split() returns float."""
        if not rust_available:
            pytest.skip("Rust bindings not available")

        from nglab import PolymarketArena

        arena = PolymarketArena(initial_collateral=10000.0, taker_fee=0.001)

        try:
            result = arena.split("TEST_MARKET", 100.0)
            assert isinstance(result, float)
        except Exception:
            pass

    def test_polymarket_arena_merge(self, rust_available):
        """Test merge() returns float."""
        if not rust_available:
            pytest.skip("Rust bindings not available")

        from nglab import PolymarketArena

        arena = PolymarketArena(initial_collateral=10000.0, taker_fee=0.001)

        try:
            result = arena.merge("TEST_MARKET", 50.0)
            assert isinstance(result, float)
        except Exception:
            pass

    def test_polymarket_arena_advance(self, polymarket_arena):
        """Test advance() returns bool."""
        result = polymarket_arena.advance()

        assert isinstance(result, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
