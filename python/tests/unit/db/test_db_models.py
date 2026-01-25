import unittest

from python.src.db.models import (
    Base,
    MarketData,
    ModelCheckpoint,
    PortfolioSnapshot,
    Trade,
)


class TestDatabaseModels(unittest.TestCase):
    def test_base_class(self):
        # Base is a DeclarativeBase, it doesn't have __tablename__
        self.assertTrue(hasattr(Base, "metadata"))

    def test_trade_model(self):
        # Test tablename
        self.assertEqual(Trade.__tablename__, "trades")

        # Test columns exist
        self.assertTrue(hasattr(Trade, "id"))
        self.assertTrue(hasattr(Trade, "timestamp"))
        self.assertTrue(hasattr(Trade, "symbol"))
        self.assertTrue(hasattr(Trade, "side"))
        self.assertTrue(hasattr(Trade, "price"))
        self.assertTrue(hasattr(Trade, "quantity"))
        self.assertTrue(hasattr(Trade, "value"))
        self.assertTrue(hasattr(Trade, "order_id"))
        self.assertTrue(hasattr(Trade, "agent_id"))
        self.assertTrue(hasattr(Trade, "extra_metadata"))

    def test_portfolio_snapshot_model(self):
        self.assertEqual(PortfolioSnapshot.__tablename__, "portfolio_snapshots")

        # Test columns exist
        self.assertTrue(hasattr(PortfolioSnapshot, "id"))
        self.assertTrue(hasattr(PortfolioSnapshot, "timestamp"))
        self.assertTrue(hasattr(PortfolioSnapshot, "agent_id"))
        self.assertTrue(hasattr(PortfolioSnapshot, "cash"))
        self.assertTrue(hasattr(PortfolioSnapshot, "position"))
        self.assertTrue(hasattr(PortfolioSnapshot, "portfolio_value"))
        self.assertTrue(hasattr(PortfolioSnapshot, "sharpe_ratio"))
        self.assertTrue(hasattr(PortfolioSnapshot, "max_drawdown"))
        self.assertTrue(hasattr(PortfolioSnapshot, "total_return"))

    def test_model_checkpoint_model(self):
        self.assertEqual(ModelCheckpoint.__tablename__, "model_checkpoints")

        # Test columns exist
        self.assertTrue(hasattr(ModelCheckpoint, "id"))
        self.assertTrue(hasattr(ModelCheckpoint, "created_at"))
        self.assertTrue(hasattr(ModelCheckpoint, "model_name"))
        self.assertTrue(hasattr(ModelCheckpoint, "version"))
        self.assertTrue(hasattr(ModelCheckpoint, "architecture"))
        self.assertTrue(hasattr(ModelCheckpoint, "hyperparameters"))
        self.assertTrue(hasattr(ModelCheckpoint, "metrics"))
        self.assertTrue(hasattr(ModelCheckpoint, "checkpoint_path"))
        self.assertTrue(hasattr(ModelCheckpoint, "git_commit"))

    def test_market_data_model(self):
        self.assertEqual(MarketData.__tablename__, "market_data")

        # Test columns exist
        self.assertTrue(hasattr(MarketData, "id"))
        self.assertTrue(hasattr(MarketData, "timestamp"))
        self.assertTrue(hasattr(MarketData, "symbol"))
        self.assertTrue(hasattr(MarketData, "source"))
        self.assertTrue(hasattr(MarketData, "bid"))
        self.assertTrue(hasattr(MarketData, "ask"))
        self.assertTrue(hasattr(MarketData, "last"))
        self.assertTrue(hasattr(MarketData, "volume"))
        self.assertTrue(hasattr(MarketData, "extra_metadata"))
