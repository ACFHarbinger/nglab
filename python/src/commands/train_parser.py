"""
Training command parser.
"""

import argparse


def add_train_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments for the training command."""
    # Model
    parser.add_argument(
        "--model", default="lstm", help="Model, 'lstm' or 'nstransformer'"
    )
    parser.add_argument(
        "--n_seq", type=int, default=1, help="Number of input sequences"
    )
    parser.add_argument(
        "--seq_len", type=int, default=21, help="Dimension of input sequence"
    )
    parser.add_argument(
        "--pred_len", type=int, default=3, help="Dimension of output sequence"
    )
    parser.add_argument(
        "--embedding_dim", type=int, default=128, help="Dimension of input embedding"
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=128,
        help="Dimension of hidden layers in Enc/Dec",
    )
    parser.add_argument(
        "--n_encode_layers",
        type=int,
        default=2,
        help="Number of layers in the encoder network",
    )

    # Training
    parser.add_argument(
        "--n_epochs", type=int, default=100, help="The number of epochs to train"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Number of instances per batch during training",
    )
    parser.add_argument(
        "--lr_model",
        type=float,
        default=1e-4,
        help="Set the learning rate for the actor network",
    )
    parser.add_argument(
        "--lr_decay", type=float, default=1.0, help="Learning rate decay per epoch"
    )
    parser.add_argument(
        "--seed", type=int, default=1234, help="Random seed to use"
    )
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA")
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Maximum L2 norm for gradient clipping, default 1.0 (0 to disable clipping)",
    )

    # Other
    parser.add_argument("--data_dir", help="Path to data directory")
    parser.add_argument(
        "--log_dir",
        default="logs",
        help="Directory to write TensorBoard information to",
    )
    parser.add_argument(
        "--run_name", default="run", help="Name to identify the run"
    )
    parser.add_argument(
        "--output_dir", default="results", help="Directory to write output models to"
    )
    parser.add_argument(
        "--load_path",
        default=None,
        help="Path to load model parameters and optimizer state from",
    )
    parser.add_argument(
        "--epoch_start",
        type=int,
        default=0,
        help="Start at epoch # (relevant for learning rate decay)",
    )
    parser.add_argument(
        "--no_tensorboard",
        action="store_true",
        help="Disable logging TensorBoard files",
    )
    parser.add_argument(
        "--no_progress_bar", action="store_true", help="Disable progress bar"
    )
    parser.add_argument(
        "--checkpoint_epochs",
        type=int,
        default=100,
        help="Save checkpoint every n epochs, 0 to save no checkpoints",
    )
    parser.add_argument(
        "--log_step", type=int, default=1, help="Log info every log_step steps"
    )
