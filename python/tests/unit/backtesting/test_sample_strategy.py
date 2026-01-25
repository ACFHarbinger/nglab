import unittest
from unittest.mock import MagicMock, patch

from python.src.backtesting.sample_strategy import SMACrossoverStrategy


class TestSMACrossoverStrategy(unittest.TestCase):
    def test_init(self):
        strategy = SMACrossoverStrategy(amount_to_buy=15.0)
        self.assertEqual(strategy.name, "SMACrossover")
        self.assertEqual(strategy.amount_to_buy, 15.0)
        self.assertIsNone(strategy.last_price)

    def test_on_market_data_no_engine(self):
        strategy = SMACrossoverStrategy()
        # Should not raise error even without engine
        strategy.on_market_data("market1", 100.0, 1234567890)

    def test_on_market_data_price_increase(self):
        strategy = SMACrossoverStrategy(amount_to_buy=10.0)
        mock_engine = MagicMock()
        strategy.set_engine(mock_engine)

        # First data point
        strategy.on_market_data("market1", 99.0, 1000)
        # Second data point (price increase)
        strategy.on_market_data("market1", 100.0, 2000)

        mock_engine.buy_yes.assert_called_once_with("market1", 10.0)

    def test_on_market_data_price_decrease(self):
        strategy = SMACrossoverStrategy(amount_to_buy=10.0)
        mock_engine = MagicMock()
        mock_engine.get_position.return_value = (20.0, 0.0)  # yes, no
        strategy.set_engine(mock_engine)

        # First data point
        strategy.on_market_data("market1", 100.0, 1000)
        # Second data point (price decrease)
        strategy.on_market_data("market1", 95.0, 2000)

        mock_engine.sell_yes.assert_called_once_with("market1", 10.0)

    def test_on_market_data_price_decrease_insufficient_position(self):
        strategy = SMACrossoverStrategy(amount_to_buy=10.0)
        mock_engine = MagicMock()
        mock_engine.get_position.return_value = (5.0, 0.0)  # yes < amount_to_buy
        strategy.set_engine(mock_engine)

        strategy.on_market_data("market1", 100.0, 1000)
        strategy.on_market_data("market1", 95.0, 2000)

        # Should not sell if position is insufficient
        mock_engine.sell_yes.assert_not_called()

    def test_on_market_data_buy_exception(self):
        strategy = SMACrossoverStrategy(amount_to_buy=10.0)
        mock_engine = MagicMock()
        mock_engine.buy_yes.side_effect = Exception("Insufficient funds")
        strategy.set_engine(mock_engine)

        with patch("builtins.print") as mock_print:
            strategy.on_market_data("market1", 99.0, 1000)
            strategy.on_market_data("market1", 100.0, 2000)

            # Should catch exception and print
            mock_print.assert_called()

    def test_on_fill(self):
        strategy = SMACrossoverStrategy()

        with patch("builtins.print") as mock_print:
            strategy.on_fill("market1", 10.0, 0.5, "buy")
            mock_print.assert_called_with("Fill: buy 10.0 @ 0.5 on market1")
