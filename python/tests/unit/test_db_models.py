"""
Tests for database models.
"""


from python.src.db.models import (
    Base,
    MarketData,
    ModelCheckpoint,
    PortfolioSnapshot,
    Trade,
)


class TestTradeModel:
    """Tests for the Trade model."""

    def test_trade_attributes(self):
        """Test that Trade model has expected columns."""
        Trade()

        # Check column names exist on the model
        assert hasattr(Trade, "id")
        assert hasattr(Trade, "timestamp")
        assert hasattr(Trade, "symbol")
        assert hasattr(Trade, "side")
        assert hasattr(Trade, "price")
        assert hasattr(Trade, "quantity")
        assert hasattr(Trade, "value")
        assert hasattr(Trade, "order_id")
        assert hasattr(Trade, "agent_id")
        assert hasattr(Trade, "extra_metadata")

    def test_trade_table_name(self):
        """Test Trade table name."""
        assert Trade.__tablename__ == "trades"

    def test_trade_constraints(self):
        """Test that Trade has proper constraints."""
        constraints = Trade.__table_args__

        # Should have check constraint for side
        constraint_names = [c.name for c in constraints if hasattr(c, "name")]
        assert "check_side" in constraint_names


class TestPortfolioSnapshotModel:
    """Tests for the PortfolioSnapshot model."""

    def test_portfolio_snapshot_attributes(self):
        """Test that PortfolioSnapshot model has expected columns."""
        assert hasattr(PortfolioSnapshot, "id")
        assert hasattr(PortfolioSnapshot, "timestamp")
        assert hasattr(PortfolioSnapshot, "agent_id")
        assert hasattr(PortfolioSnapshot, "cash")
        assert hasattr(PortfolioSnapshot, "position")
        assert hasattr(PortfolioSnapshot, "portfolio_value")
        assert hasattr(PortfolioSnapshot, "sharpe_ratio")
        assert hasattr(PortfolioSnapshot, "max_drawdown")
        assert hasattr(PortfolioSnapshot, "total_return")

    def test_portfolio_snapshot_table_name(self):
        """Test PortfolioSnapshot table name."""
        assert PortfolioSnapshot.__tablename__ == "portfolio_snapshots"


class TestModelCheckpointModel:
    """Tests for the ModelCheckpoint model."""

    def test_model_checkpoint_attributes(self):
        """Test that ModelCheckpoint model has expected columns."""
        assert hasattr(ModelCheckpoint, "id")
        assert hasattr(ModelCheckpoint, "created_at")
        assert hasattr(ModelCheckpoint, "model_name")
        assert hasattr(ModelCheckpoint, "version")
        assert hasattr(ModelCheckpoint, "architecture")
        assert hasattr(ModelCheckpoint, "hyperparameters")
        assert hasattr(ModelCheckpoint, "metrics")
        assert hasattr(ModelCheckpoint, "checkpoint_path")
        assert hasattr(ModelCheckpoint, "git_commit")

    def test_model_checkpoint_table_name(self):
        """Test ModelCheckpoint table name."""
        assert ModelCheckpoint.__tablename__ == "model_checkpoints"


class TestMarketDataModel:
    """Tests for the MarketData model."""

    def test_market_data_attributes(self):
        """Test that MarketData model has expected columns."""
        assert hasattr(MarketData, "id")
        assert hasattr(MarketData, "timestamp")
        assert hasattr(MarketData, "symbol")
        assert hasattr(MarketData, "source")
        assert hasattr(MarketData, "bid")
        assert hasattr(MarketData, "ask")
        assert hasattr(MarketData, "last")
        assert hasattr(MarketData, "volume")
        assert hasattr(MarketData, "extra_metadata")

    def test_market_data_table_name(self):
        """Test MarketData table name."""
        assert MarketData.__tablename__ == "market_data"


class TestBaseModel:
    """Tests for the Base declarative model."""

    def test_base_is_declarative(self):
        """Test that Base is a valid declarative base."""
        assert hasattr(Base, "metadata")

    def test_models_inherit_from_base(self):
        """Test that all models inherit from Base."""
        assert issubclass(Trade, Base)
        assert issubclass(PortfolioSnapshot, Base)
        assert issubclass(ModelCheckpoint, Base)
        assert issubclass(MarketData, Base)
