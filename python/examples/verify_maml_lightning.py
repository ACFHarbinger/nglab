"""
Verification script for MAML Lightning Module.
"""

import pytorch_lightning as pl
import torch

from python.src.models.time_series import TimeSeriesBackbone
from python.src.pipeline.meta.maml import MAMLDataModule, MAMLLightningModule


def verify_maml_lightning():
    print("--- MAML Lightning Module Verification ---")

    # Create base model
    config = {
        "name": "LSTM",
        "feature_dim": 5,
        "hidden_dim": 32,
        "num_layers": 1,
        "output_dim": 1,
        "seq_len": 10,
        "output_type": "prediction",
    }
    base_model = TimeSeriesBackbone(config)

    # Create MAML Lightning module
    maml = MAMLLightningModule(
        model=base_model,
        inner_lr=0.01,
        outer_lr=0.001,
        inner_steps=3,
        meta_batch_size=4,
    )
    print(
        f"Created MAMLLightningModule with {sum(p.numel() for p in maml.parameters())} parameters"
    )

    # Create synthetic regime datasets
    print("\nCreating synthetic regime datasets...")
    # Shape: [num_samples, seq_len, features]
    # For seq_len=10,features=5, we'll create flattened data then reshape
    regime_datasets = {
        0: torch.randn(
            200, 10, 6
        ),  # Regime 0: 200 samples, seq_len=10, 5 features + 1 target
        1: torch.randn(200, 10, 6),  # Regime 1
        2: torch.randn(200, 10, 6),  # Regime 2
    }

    # Create DataModule
    datamodule = MAMLDataModule(
        regime_datasets=regime_datasets,
        support_size=20,
        query_size=10,
        meta_batch_size=4,
    )

    # Create trainer (limit to 2 epochs for quick verification)
    print("\nStarting meta-training...")
    trainer = pl.Trainer(
        max_epochs=2,
        accelerator="cpu",
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=True,
        num_sanity_val_steps=0,  # Skip sanity check
    )

    # Meta-train
    trainer.fit(maml, datamodule)

    print("\n✓ MAML Lightning verification passed!")

    # Test fast adaptation
    print("\nTesting fast adaptation...")
    support_x = torch.randn(20, 10, 5)
    support_y = torch.randn(20, 1)

    adapted_model = maml.adapt(support_x, support_y, num_steps=5)

    # Test adapted model
    test_x = torch.randn(5, 10, 5)
    with torch.no_grad():
        pred = adapted_model(test_x)
    print(f"Adapted model prediction shape: {pred.shape}")

    print("\n✓ Fast adaptation test passed!")


if __name__ == "__main__":
    verify_maml_lightning()
