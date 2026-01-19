"""
Hyperparameter Optimization (HPO) command parser.
"""

import argparse


def add_hpo_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments for the hp_optim (HPO) command."""
    parser.add_argument(
        "--model", default="lstm", help="Model to optimize, 'lstm' or 'nstransformer'"
    )
    parser.add_argument(
        "--num_samples", type=int, default=10, help="Number of HPO samples to run"
    )
    parser.add_argument(
        "--max_epochs", type=int, default=10, help="Maximum epochs per trial"
    )
    parser.add_argument(
        "--gpus_per_trial", type=float, default=0.0, help="GPUs to allocate per trial"
    )
    parser.add_argument(
        "--data_dir", help="Path to data directory"
    )
    parser.add_argument(
        "--seed", type=int, default=1234, help="Random seed"
    )
