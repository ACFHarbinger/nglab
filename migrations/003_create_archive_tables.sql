-- Data Archival Tables for NGLab
-- Archive old training runs and predictions to reduce main table size

-- Archive table for old training runs (30+ days)
CREATE TABLE IF NOT EXISTS training_runs_archive (
    LIKE training_runs INCLUDING ALL
);

ALTER TABLE training_runs_archive 
ADD COLUMN IF NOT EXISTS archived_at TIMESTAMP DEFAULT NOW();

COMMENT ON TABLE training_runs_archive IS 
'Archived training runs older than 30 days';

-- Archive table for old predictions
CREATE TABLE IF NOT EXISTS predictions_archive (
    LIKE predictions INCLUDING ALL
);

ALTER TABLE predictions_archive 
ADD COLUMN IF NOT EXISTS archived_at TIMESTAMP DEFAULT NOW();

COMMENT ON TABLE predictions_archive IS 
'Archived predictions older than 30 days';

-- Archive table for old metrics
CREATE TABLE IF NOT EXISTS metrics_archive (
    LIKE metrics INCLUDING ALL
);

ALTER TABLE metrics_archive 
ADD COLUMN IF NOT EXISTS archived_at TIMESTAMP DEFAULT NOW();

-- Index on archived_at for retention policy queries
CREATE INDEX IF NOT EXISTS idx_training_runs_archive_archived 
ON training_runs_archive(archived_at);

CREATE INDEX IF NOT EXISTS idx_predictions_archive_archived 
ON predictions_archive(archived_at);

CREATE INDEX IF NOT EXISTS idx_metrics_archive_archived 
ON metrics_archive(archived_at);

-- Function to archive old training runs
CREATE OR REPLACE FUNCTION archive_old_training_runs(days_old INTEGER DEFAULT 30)
RETURNS INTEGER AS $$
DECLARE
    rows_archived INTEGER;
BEGIN
    WITH moved_rows AS (
        DELETE FROM training_runs
        WHERE created_at < NOW() - (days_old || ' days')::INTERVAL
        RETURNING *
    )
    INSERT INTO training_runs_archive
    SELECT *, NOW() as archived_at FROM moved_rows;
    
    GET DIAGNOSTICS rows_archived = ROW_COUNT;
    RETURN rows_archived;
END;
$$ LANGUAGE plpgsql;

-- Function to archive old predictions
CREATE OR REPLACE FUNCTION archive_old_predictions(days_old INTEGER DEFAULT 30)
RETURNS INTEGER AS $$
DECLARE
    rows_archived INTEGER;
BEGIN
    WITH moved_rows AS (
        DELETE FROM predictions
        WHERE timestamp < NOW() - (days_old || ' days')::INTERVAL
        RETURNING *
    )
    INSERT INTO predictions_archive
    SELECT *, NOW() as archived_at FROM moved_rows;
    
    GET DIAGNOSTICS rows_archived = ROW_COUNT;
    RETURN rows_archived;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION archive_old_training_runs IS 
'Archives training runs older than specified days (default 30)';

COMMENT ON FUNCTION archive_old_predictions IS 
'Archives predictions older than specified days (default 30)';
