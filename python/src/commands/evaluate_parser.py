
from __future__ import annotations

from argparse import ArgumentParser, _SubParsersAction


def add_evaluate_args(parser: ArgumentParser | _SubParsersAction) -> None:
    """Add arguments for the evaluate command."""
    if isinstance(parser, _SubParsersAction):
        parser = parser.add_parser("evaluate", help="Evaluate model performance")

    parser.add_argument(
        "--load-path",
        type=str,
        required=True,
        help="Path to the model checkpoint to evaluate",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/polymarket/",
        help="Directory containing evaluation data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Data split to evaluate on",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualization plots",
    )
