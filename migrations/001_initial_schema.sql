-- Trade history
CREATE TABLE trades (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(4) NOT NULL CHECK (side IN ('buy', 'sell')),
    price NUMERIC(20, 8) NOT NULL,
    quantity NUMERIC(20, 8) NOT NULL,
    value NUMERIC(20, 8) NOT NULL,
    order_id UUID NOT NULL,
    agent_id VARCHAR(50),
    extra_metadata JSONB
);

CREATE INDEX idx_trades_timestamp ON trades(timestamp DESC);
CREATE INDEX idx_trades_symbol ON trades(symbol);
CREATE INDEX idx_trades_agent_id ON trades(agent_id);

-- Portfolio snapshots
CREATE TABLE portfolio_snapshots (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    agent_id VARCHAR(50) NOT NULL,
    cash NUMERIC(20, 8) NOT NULL,
    position NUMERIC(20, 8) NOT NULL,
    portfolio_value NUMERIC(20, 8) NOT NULL,
    sharpe_ratio NUMERIC(10, 4),
    max_drawdown NUMERIC(10, 4),
    total_return NUMERIC(10, 4)
);

CREATE INDEX idx_portfolio_timestamp ON portfolio_snapshots(timestamp DESC);
CREATE INDEX idx_portfolio_agent_id ON portfolio_snapshots(agent_id);

-- Model checkpoints
CREATE TABLE model_checkpoints (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    model_name VARCHAR(100) NOT NULL,
    version VARCHAR(50) NOT NULL,
    architecture TEXT NOT NULL,
    hyperparameters JSONB NOT NULL,
    metrics JSONB NOT NULL,
    checkpoint_path TEXT NOT NULL,
    git_commit VARCHAR(40),
    UNIQUE(model_name, version)
);

-- Market data
CREATE TABLE market_data (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    source VARCHAR(50) NOT NULL,
    bid NUMERIC(20, 8),
    ask NUMERIC(20, 8),
    last NUMERIC(20, 8),
    volume NUMERIC(20, 8),
    extra_metadata JSONB
);

CREATE INDEX idx_market_data_symbol_timestamp ON market_data(symbol, timestamp DESC);
