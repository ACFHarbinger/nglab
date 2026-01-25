import json
from unittest.mock import MagicMock, patch

import pytest

from python.src.backtesting.engine import BacktestEngine
from python.src.backtesting.strategy import BaseStrategy


class MockStrategy(BaseStrategy):
    def on_market_data(self, market_id: str, price: float, timestamp: int) -> None:
        pass

    def on_fill(self, market_id: str, amount: float, price: float, side: str) -> None:
        pass


@pytest.fixture
def mock_arena():
    with patch("python.src.backtesting.engine.PolymarketArena") as mock:
        instance = mock.return_value
        instance.current_step.return_value = 0
        instance.collateral.return_value = 10000.0
        instance.account_value.return_value = 10000.0
        instance.realized_pnl.return_value = 0.0
        instance.advance.side_effect = [True, False]  # Run for 2 ticks
        instance.get_price.return_value = 0.5
        yield instance


def test_backtest_engine_init(mock_arena):
    engine = BacktestEngine(initial_collateral=5000.0, taker_fee=0.002)
    # Check if PolymarketArena was initialized with correct args
    from python.src.backtesting.engine import PolymarketArena

    PolymarketArena.assert_called_once_with(5000.0, 0.002)
    assert engine.strategy is None


def test_backtest_engine_set_strategy(mock_arena):
    engine = BacktestEngine()
    strategy = MockStrategy()
    engine.set_strategy(strategy)
    assert engine.strategy == strategy
    assert strategy.engine == engine


def test_backtest_engine_load_data(mock_arena):
    engine = BacktestEngine()
    markets_json = json.dumps([{"id": "M1"}, {"id": "M2"}])
    price_histories = {"M1": "csv_data_1", "M2": "csv_data_2"}

    engine.load_data(markets_json, price_histories)

    mock_arena.load_markets.assert_called_once_with(markets_json)
    assert engine.market_ids == ["M1", "M2"]
    assert mock_arena.load_price_history.call_count == 2


def test_backtest_engine_run(mock_arena):
    engine = BacktestEngine()
    strategy = MockStrategy()
    strategy.on_market_data = MagicMock()
    engine.set_strategy(strategy)
    engine.market_ids = ["M1"]

    history = engine.run()

    assert len(history) == 2
    assert strategy.on_market_data.call_count == 2
    mock_arena.advance.assert_called()


def test_backtest_engine_run_no_strategy(mock_arena):
    engine = BacktestEngine()
    with pytest.raises(ValueError, match="No strategy assigned"):
        engine.run()


def test_backtest_engine_buy_yes(mock_arena):
    engine = BacktestEngine()
    strategy = MockStrategy()
    strategy.on_fill = MagicMock()
    engine.set_strategy(strategy)

    mock_arena.buy_yes.return_value = 100.0
    cost = engine.buy_yes("M1", 50.0)

    assert cost == 100.0
    mock_arena.buy_yes.assert_called_once_with("M1", 50.0)
    strategy.on_fill.assert_called_once_with("M1", 50.0, 0.5, "buy_yes")


def test_backtest_engine_buy_no(mock_arena):
    engine = BacktestEngine()
    strategy = MockStrategy()
    strategy.on_fill = MagicMock()
    engine.set_strategy(strategy)

    mock_arena.buy_no.return_value = 80.0
    cost = engine.buy_no("M1", 50.0)

    assert cost == 80.0
    mock_arena.buy_no.assert_called_once_with("M1", 50.0)
    # Price is 0.5, so 1.0 - 0.5 = 0.5
    strategy.on_fill.assert_called_once_with("M1", 50.0, 0.5, "buy_no")


def test_backtest_engine_sell_yes(mock_arena):
    engine = BacktestEngine()
    strategy = MockStrategy()
    strategy.on_fill = MagicMock()
    engine.set_strategy(strategy)

    mock_arena.sell_yes.return_value = 60.0
    proceeds = engine.sell_yes("M1", 50.0)

    assert proceeds == 60.0
    mock_arena.sell_yes.assert_called_once_with("M1", 50.0)
    strategy.on_fill.assert_called_once_with("M1", 50.0, 0.5, "sell_yes")


def test_backtest_engine_split_merge(mock_arena):
    engine = BacktestEngine()
    mock_arena.split.return_value = 10.0
    mock_arena.merge.return_value = 20.0

    assert engine.split("M1", 5.0) == 10.0
    assert engine.merge("M1", 5.0) == 20.0
    mock_arena.split.assert_called_once_with("M1", 5.0)
    mock_arena.merge.assert_called_once_with("M1", 5.0)


def test_backtest_engine_getters(mock_arena):
    engine = BacktestEngine()
    mock_arena.get_position.return_value = (10.0, 5.0)
    mock_arena.get_price.return_value = 0.7

    assert engine.get_position("M1") == (10.0, 5.0)
    assert engine.get_price("M1") == 0.7
    mock_arena.get_position.assert_called_once_with("M1")
    mock_arena.get_price.assert_called_with("M1")
