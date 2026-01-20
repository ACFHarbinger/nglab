"""
Supervised Learning Module for NGLab.

Implements standard supervised fine-tuning of pre-trained backbones
for specific downstream prediction tasks.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, cast

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn
from torch.utils.data import DataLoader

from .base import BaseModule


class SLLightningModule(BaseModule):
    """
    Module for Supervised Learning (Fine-tuning).
    """

    def __init__(self, backbone: nn.Module, cfg: Dict[str, Any]) -> None:
        """
        Initialize the Supervised module.

        Args:
            backbone (nn.Module): The pre-trained time-series model backbone.
            cfg (Dict): Configuration parameters.
        """
        super().__init__(cfg)
        self.save_hyperparameters(ignore=["backbone"])
        self.backbone = backbone
        
        # Determine output dim
        hidden_dim = int(cfg.get("hidden_dim", 128))
        output_dim = int(cfg.get("output_dim", 1))
        
        self.head = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the backbone and head.
        """
        feat = self.backbone(x)
        return cast(torch.Tensor, self.head(feat))

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """
        Perform a supervised training step.
        """
        # Batch: {observation, target}
        if isinstance(batch, dict):
             x = batch["observation"]
             y = batch["target"]
        else:
             x, y = batch

        pred = self(x)
        loss = F.mse_loss(pred, y)  # Or CrossEntropy relative to task
        loss.backward()  # type: ignore

        self.log("train/sl_loss", loss)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """
        Perform a validation step.
        """
        if isinstance(batch, dict):
             x = batch["observation"]
             y = batch["target"]
        else:
             x, y = batch
             
        pred = self(x)
        loss = F.mse_loss(pred, y)
        self.log("val/sl_loss", loss)
        return loss


class ProgressCallback:
    """Callback to stream training progress as JSON."""

    def __init__(self, total_epochs: int) -> None:
        """
        Initialize the progress callback.

        Args:
            total_epochs (int): Total number of epochs for progress calculation.
        """
        self.total_epochs = total_epochs

    def on_epoch_end(
        self, epoch: int, train_loss: float, val_loss: Optional[float] = None
    ) -> None:
        """Emit progress JSON to stdout."""
        progress = {
            "type": "progress",
            "epoch": epoch + 1,
            "total_epochs": self.total_epochs,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6) if val_loss is not None else None,
            "percent": round((epoch + 1) / self.total_epochs * 100, 1),
        }
        print(json.dumps(progress), flush=True)


def train_from_csv(  # noqa: PLR0913, PLR0915
    csv_path: str,
    target_column: str,
    model_name: str,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    seq_len: int = 30,
    pred_len: int = 1,
    train_split: float = 0.8,
    model_params: Optional[Dict[str, Any]] = None,
    output_path: Optional[str] = None,
) -> str:
    """
    Train a supervised model from CSV data.

    Args:
        csv_path: Path to CSV file.
        target_column: Column to predict.
        model_name: Model architecture name.
        epochs: Number of training epochs.
        batch_size: Training batch size.
        learning_rate: Learning rate.
        seq_len: Input sequence length.
        pred_len: Prediction length.
        train_split: Train/validation split ratio.
        model_params: Additional model parameters.
        output_path: Path to save trained model.

    Returns:
        Path to saved model.
    """
    # Add src to path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from utils.model_versioning import ModelMetadata, save_model_with_metadata  # type: ignore

    from data.time_series_dataset import TimeSeriesDataset  # type: ignore
    from models.time_series import TimeSeriesBackbone  # type: ignore

    # Create datasets
    train_dataset = TimeSeriesDataset(
        csv_path=csv_path,
        target_column=target_column,
        seq_len=seq_len,
        pred_len=pred_len,
        train=True,
        train_ratio=train_split,
    )
    # Pass training stats to validation set to ensure consistent scaling
    train_stats = {
        "min": train_dataset.raw_min,
        "max": train_dataset.raw_max,
        "mean": train_dataset.raw_mean,
        "std": train_dataset.raw_std,
    }
    val_dataset = TimeSeriesDataset(
        csv_path=csv_path,
        target_column=target_column,
        seq_len=seq_len,
        pred_len=pred_len,
        train=False,
        train_ratio=train_split,
        stats=train_stats,
    )

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Build model config
    cfg: Dict[str, Any] = {
        "name": model_name,
        "feature_dim": 1,  # Univariate for now
        "hidden_dim": int(model_params.get("hidden_dim", 128)) if model_params else 128,
        "num_layers": int(model_params.get("n_layers", 2)) if model_params else 2,
        "d_model": int(model_params.get("d_model", 128)) if model_params else 128,
        "num_heads": int(model_params.get("num_heads", 4)) if model_params else 4,
        "seq_len": int(seq_len),
        "pred_len": int(pred_len),
        "output_dim": int(pred_len),
        "dropout": float(model_params.get("dropout", 0.1)) if model_params else 0.1,
        "output_type": "prediction",
        "normalization": {
            "method": train_dataset.normalize,
            "min": train_dataset.raw_min,
            "max": train_dataset.raw_max,
            "mean": train_dataset.raw_mean,
            "std": train_dataset.raw_std,
        },
    }

    # Add any extra model params with intelligent casting
    if model_params:
        for k, v in model_params.items():
            if k not in cfg:
                # Try to cast to int/float if it's a string
                if isinstance(v, str):
                    try:
                        if "." in v:
                            cfg[k] = float(v)
                        else:
                            cfg[k] = int(v)
                    except ValueError:
                        cfg[k] = v
                else:
                    cfg[k] = v

    # Create model (using the backbone's built-in prediction head)
    model = TimeSeriesBackbone(cfg)

    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Progress callback
    progress = ProgressCallback(epochs)

    # Training loop
    best_val_loss = float("inf")
    best_model_state = None
    avg_train_loss = 0.0
    avg_val_loss = 0.0

    for epoch in range(epochs):
        # Train
        model.train()
        train_losses = []
        for batch in train_loader:
            x = batch["observation"].to(device)
            y = batch["target"].to(device)

            optimizer.zero_grad()
            pred = model(x)  # [B, pred_len]
            # y is already [B, pred_len] from dataset collate
            loss = F.mse_loss(pred, y)
            loss.backward()  # type: ignore
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = sum(train_losses) / len(train_losses) if train_losses else 0

        # Validate
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                x = batch["observation"].to(device)
                y = batch["target"].to(device)
                pred = model(x)
                loss = F.mse_loss(pred, y)
                val_losses.append(loss.item())

        avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else 0.0

        # Track best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            import copy

            best_model_state = copy.deepcopy(model.state_dict())

        # Step scheduler
        scheduler.step()

        # Emit progress
        progress.on_epoch_end(epoch, avg_train_loss, avg_val_loss)

    # Load best state before saving
    if best_model_state:
        model.load_state_dict(best_model_state)

    # Determine output path
    if output_path is None:
        output_dir = Path(__file__).parent.parent.parent.parent.parent / "model_weights"
        output_dir.mkdir(exist_ok=True)
        output_path_resolved = str(
            output_dir / f"{model_name.lower()}_{target_column}.pt"
        )
    else:
        output_path_resolved = output_path

    # Save model with metadata
    metadata = ModelMetadata(
        version="1.0.0",
        model_type=model_name,
        framework_version=f"pytorch-{torch.__version__}",
        hyperparameters=cfg,
        metrics={
            "final_train_loss": avg_train_loss,
            "final_val_loss": avg_val_loss if avg_val_loss is not None else 0.0,
        },
        training_date=datetime.now().isoformat(),
        dataset_hash=f"csv:{Path(csv_path).name}",
        description=f"Supervised training on {target_column}",
    )

    save_model_with_metadata(model, output_path_resolved, metadata)

    # Emit completion
    result = {
        "type": "complete",
        "status": "success",
        "model_path": output_path_resolved,
        "final_train_loss": avg_train_loss,
        "final_val_loss": avg_val_loss,
    }
    print(json.dumps(result), flush=True)

    return str(output_path_resolved)


def main() -> None:
    """CLI entry point for supervised training."""
    parser = argparse.ArgumentParser(description="Train supervised time series model")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to CSV file")
    parser.add_argument(
        "--target_column", type=str, required=True, help="Column to predict"
    )
    parser.add_argument(
        "--model_name", type=str, default="LSTM", help="Model architecture"
    )
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="Learning rate"
    )
    parser.add_argument("--seq_len", type=int, default=30, help="Sequence length")
    parser.add_argument("--pred_len", type=int, default=1, help="Prediction length")
    parser.add_argument(
        "--train_split", type=float, default=0.8, help="Train/val split"
    )
    parser.add_argument(
        "--model_params", type=str, default="{}", help="JSON model params"
    )
    parser.add_argument("--output_path", type=str, default=None, help="Output path")
    parser.add_argument(
        "--list_columns", action="store_true", help="List CSV columns and exit"
    )

    args = parser.parse_args()

    # Handle list columns mode
    if args.list_columns:
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from data.time_series_dataset import TimeSeriesDataset

        columns = TimeSeriesDataset.list_columns(args.csv_path)
        result = {"type": "columns", "columns": columns}
        print(json.dumps(result))
        return

    try:
        model_params = json.loads(args.model_params)
    except json.JSONDecodeError:
        model_params = {}

    try:
        train_from_csv(
            csv_path=args.csv_path,
            target_column=args.target_column,
            model_name=args.model_name,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            train_split=args.train_split,
            model_params=model_params,
            output_path=args.output_path,
        )
    except Exception as e:
        error = {"type": "error", "status": "error", "message": str(e)}
        print(json.dumps(error), flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
