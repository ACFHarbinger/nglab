"""
Active learning command parser.
"""

import argparse


def add_active_learning_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments for the active-learning command."""
    parser.add_argument(
        "--method",
        choices=["quantile", "mc_dropout", "random"],
        default="quantile",
        help="Uncertainty estimation method.",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=10,
        help="Number of samples to select from the pool.",
    )
    parser.add_argument(
        "--pool_path", type=str, help="Path to the unlabelled data pool (CSV)."
    )
