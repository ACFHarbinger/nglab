"""
Fixtures for nglab Rust bindings testing.
"""

import pytest


@pytest.fixture(scope="session")
def rust_available() -> bool:
    """Check if nglab Rust bindings are available."""
    try:
        import nglab  # noqa: F401
        from nglab import Arena, OrderBook, PolymarketArena, TradingEnv  # noqa: F401

        return True
    except (ImportError, AttributeError):
        return False


@pytest.fixture
def arena_instance(rust_available):
    """Initialized Arena instance (skip if Rust unavailable)."""
    if not rust_available:
        pytest.skip("Rust bindings not available")

    from nglab import Arena

    return Arena()


@pytest.fixture
def orderbook_instance(rust_available):
    """Initialized OrderBook instance."""
    if not rust_available:
        pytest.skip("Rust bindings not available")

    from nglab import OrderBook

    return OrderBook()


@pytest.fixture
def rust_trading_env(rust_available):
    """Initialized Rust TradingEnv with sample prices."""
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

    # Load sample prices
    prices = [100.0 + i * 0.5 for i in range(200)]
    env.load_prices(prices)

    return env


@pytest.fixture
def polymarket_arena(rust_available):
    """Initialized PolymarketArena instance."""
    if not rust_available:
        pytest.skip("Rust bindings not available")

    from nglab import PolymarketArena

    arena = PolymarketArena(
        initial_collateral=10000.0,
        taker_fee=0.001,
    )

    return arena
