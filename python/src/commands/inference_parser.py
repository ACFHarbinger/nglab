"""
Inference command parser.
"""

import argparse


def add_inference_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments for the inference command."""
    parser.add_argument(
        "--model", default="lstm", help="Model, 'lstm' or 'nstransformer'"
    )
    parser.add_argument("--data_dir", help="Path to data directory")
    parser.add_argument(
        "--load_path", help="Path to load model parameters and optimizer state from"
    )
    parser.add_argument("--id", help="ID of time series to predict")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed to use")
