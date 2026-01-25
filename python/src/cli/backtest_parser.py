
from __future__ import annotations

from argparse import ArgumentParser, _SubParsersAction


def add_backtest_args(parser: ArgumentParser | _SubParsersAction) -> None:
    """Add arguments for the backtest command."""
    if isinstance(parser, _SubParsersAction):
        parser = parser.add_parser("backtest", help="Run historical backtest")

    parser.add_argument(
        "--strategy",
        type=str,
        required=True,
        help="Name of the strategy to backtest",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date for backtest (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        help="End date for backtest (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=10000.0,
        help="Initial capital for the backtest",
    )
    parser.add_argument(
        "--data-source",
        type=str,
        default="polymarket",
        help="Data source name",
    )
