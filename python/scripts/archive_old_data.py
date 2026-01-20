#!/usr/bin/env python3
"""
Data archival script for NGLab.

Archives old training runs and predictions to reduce database size
while preserving historical data.

Usage:
    python scripts/archive_old_data.py [--days=30] [--dry-run]
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime

import asyncpg

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def archive_data(days_old: int = 30, dry_run: bool = False) -> dict[str, int]:
    """
    Archive old data from main tables to archive tables.

    Args:
        days_old: Archive data older than this many days
        dry_run: If True, only count rows without archiving

    Returns:
        Dictionary with archival counts per table
    """
    # Database connection
    conn = await asyncpg.connect(
        host="postgres",
        port=5432,
        database="nglab",
        user="nglab",
        password="nglab_password",  # Should come from env in production
    )

    results = {}

    try:
        # Archive training runs
        if dry_run:
            count = await conn.fetchval(
                "SELECT COUNT(*) FROM training_runs WHERE created_at < NOW() - $1::INTERVAL",
                f"{days_old} days",
            )
            logger.info(f"Would archive {count} training runs")
            results["training_runs"] = count
        else:
            count = await conn.fetchval(
                "SELECT archive_old_training_runs($1)", days_old
            )
            logger.info(f"Archived {count} training runs")
            results["training_runs"] = count

        # Archive predictions
        if dry_run:
            count = await conn.fetchval(
                "SELECT COUNT(*) FROM predictions WHERE timestamp < NOW() - $1::INTERVAL",
                f"{days_old} days",
            )
            logger.info(f"Would archive {count} predictions")
            results["predictions"] = count
        else:
            count = await conn.fetchval("SELECT archive_old_predictions($1)", days_old)
            logger.info(f"Archived {count} predictions")
            results["predictions"] = count

        # Vacuum to reclaim space
        if not dry_run:
            logger.info("Running VACUUM ANALYZE...")
            await conn.execute("VACUUM ANALYZE training_runs, predictions")
            logger.info("VACUUM complete")

        return results

    finally:
        await conn.close()


async def main():
    parser = argparse.ArgumentParser(description="Archive old NGLab data")
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Archive data older than this many days (default: 30)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Only count rows without archiving"
    )

    args = parser.parse_args()

    logger.info(f"Starting archival (days_old={args.days}, dry_run={args.dry_run})")
    start_time = datetime.now()

    try:
        results = await archive_data(args.days, args.dry_run)

        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"Archival complete in {elapsed:.2f}s")
        logger.info(f"Summary: {results}")

        return 0

    except Exception as e:
        logger.error(f"Archival failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
