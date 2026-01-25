"""Database models for NGLab."""

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Column,
    DateTime,
    Index,
    Numeric,
    String,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Base class for all database models."""

    pass


class Trade(Base):
    """Model for trading activity records."""

    __tablename__ = "trades"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    symbol = Column(String(20), nullable=False)
    side = Column(String(4), nullable=False)
    price = Column(Numeric(20, 8), nullable=False)
    quantity = Column(Numeric(20, 8), nullable=False)
    value = Column(Numeric(20, 8), nullable=False)
    order_id = Column(UUID(as_uuid=True), nullable=False)
    agent_id = Column(String(50))
    extra_metadata = Column(JSONB)

    __table_args__ = (
        CheckConstraint("side IN ('buy', 'sell')", name="check_side"),
        Index("idx_trades_timestamp", "timestamp", postgresql_using="btree"),
        Index("idx_trades_symbol", "symbol"),
        Index("idx_trades_agent_id", "agent_id"),
    )


class PortfolioSnapshot(Base):
    """Model for agent portfolio state snapshots."""

    __tablename__ = "portfolio_snapshots"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    agent_id = Column(String(50), nullable=False)
    cash = Column(Numeric(20, 8), nullable=False)
    position = Column(Numeric(20, 8), nullable=False)
    portfolio_value = Column(Numeric(20, 8), nullable=False)
    sharpe_ratio = Column(Numeric(10, 4))
    max_drawdown = Column(Numeric(10, 4))
    total_return = Column(Numeric(10, 4))

    __table_args__ = (
        Index("idx_portfolio_timestamp", "timestamp", postgresql_using="btree"),
        Index("idx_portfolio_agent_id", "agent_id"),
    )


class ModelCheckpoint(Base):
    """Model for tracking trained model checkpoints."""

    __tablename__ = "model_checkpoints"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    created_at = Column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow
    )
    model_name = Column(String(100), nullable=False)
    version = Column(String(50), nullable=False)
    architecture = Column(String, nullable=False)
    hyperparameters = Column(JSONB, nullable=False)
    metrics = Column(JSONB, nullable=False)
    checkpoint_path = Column(String, nullable=False)
    git_commit = Column(String(40))

    __table_args__ = (
        Index(
            "idx_model_checkpoints_name_version", "model_name", "version", unique=True
        ),
    )


class MarketData(Base):
    """Model for historical market data storage."""

    __tablename__ = "market_data"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    symbol = Column(String(20), nullable=False)
    source = Column(String(50), nullable=False)
    bid = Column(Numeric(20, 8))
    ask = Column(Numeric(20, 8))
    last = Column(Numeric(20, 8))
    volume = Column(Numeric(20, 8))
    extra_metadata = Column(JSONB)

    __table_args__ = (
        Index(
            "idx_market_data_symbol_timestamp",
            "symbol",
            "timestamp",
            postgresql_using="btree",
        ),
    )
