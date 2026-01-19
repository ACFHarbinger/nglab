-- Database Index Optimization for NGLab
-- Improves query performance for common access patterns

-- Training runs and model lookups
CREATE INDEX IF NOT EXISTS idx_training_runs_user_created 
ON training_runs(user_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_models_lookup 
ON models(model_id, status) WHERE status != 'deleted';

-- Time-series data optimization
CREATE INDEX IF NOT EXISTS idx_predictions_timestamp 
ON predictions(timestamp DESC, model_id);

CREATE INDEX IF NOT EXISTS idx_metrics_timestamp 
ON metrics(timestamp DESC, experiment_id);

-- Portfolio and trading activity
CREATE INDEX IF NOT EXISTS idx_trades_user_timestamp 
ON trades(user_id, executed_at DESC);

CREATE INDEX IF NOT EXISTS idx_positions_active 
ON positions(user_id, status) WHERE status = 'open';

-- Partial index for active sessions only
CREATE INDEX IF NOT EXISTS idx_sessions_active 
ON user_sessions(user_id, expires_at) WHERE expires_at > NOW();

-- Composite index for vault secrets
CREATE INDEX IF NOT EXISTS idx_vault_secrets_user_label 
ON vault_secrets(user_id, label);

-- BRIN index for large time-series tables (space-efficient)
CREATE INDEX IF NOT EXISTS idx_orderbook_snapshots_brin 
ON orderbook_snapshots USING BRIN(timestamp);

-- Performance: Analyze tables after index creation
ANALYZE training_runs;
ANALYZE models;
ANALYZE predictions;
ANALYZE metrics;
ANALYZE trades;
ANALYZE positions;
ANALYZE user_sessions;
ANALYZE vault_secrets;

-- Add index usage tracking comment
COMMENT ON INDEX idx_training_runs_user_created IS 
'Optimizes user activity queries and dashboard loads';

COMMENT ON INDEX idx_predictions_timestamp IS 
'Optimizes time-series queries for prediction API';
