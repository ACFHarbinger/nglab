"""
Example script for training a Variational Auto-Encoder on time series data.

This script demonstrates:
1. Loading time series data
2. Setting up the VAE model
3. Training with PyTorch Lightning
4. Generating samples from the learned latent space
5. Evaluating reconstruction quality
"""

import sys
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, random_split

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipeline.lightning.vae_module import VAELightningModule

from data.polymarket_dataset import PolymarketDataset


def main():  # noqa: PLR0915
    """Main training function."""

    # ========== Configuration ==========
    # Data configuration
    DATA_DIR = Path("data/polymarket")  # Update this path
    BATCH_SIZE = 64
    NUM_WORKERS = 4
    TRAIN_SPLIT = 0.8

    # Model configuration
    INPUT_DIM = 5  # OHLCV data
    LATENT_DIM = 32
    D_MODEL = 128
    SEQ_LEN = 100
    PRED_LEN = 20
    ENCODER_TYPE = "mamba"  # Options: transformer, mamba, lstm, gru, xlstm
    N_LAYERS = 3

    # Training configuration
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5
    KL_WEIGHT = 1.0  # Beta parameter (1.0 = standard VAE, >1.0 = beta-VAE)
    KL_ANNEAL_EPOCHS = 10  # Gradually increase KL weight for stable training
    RECONSTRUCTION_LOSS = "mse"  # Options: mse, l1, huber
    MAX_EPOCHS = 100

    # Trainer configuration
    ACCELERATOR = "auto"  # 'auto', 'gpu', 'cpu'
    DEVICES = 1
    PRECISION = 32  # 16 for mixed precision

    # Seed for reproducibility
    pl.seed_everything(42)

    # ========== Load Data ==========
    print("Loading dataset...")
    try:
        dataset = PolymarketDataset(
            data_dir=str(DATA_DIR), seq_len=SEQ_LEN, pred_len=PRED_LEN
        )
        print(f"Dataset size: {len(dataset)} samples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Using synthetic data for demonstration...")
        # Create synthetic dataset for demonstration
        from torch.utils.data import TensorDataset

        num_samples = 1000
        synthetic_prices = torch.randn(num_samples, SEQ_LEN, INPUT_DIM)
        synthetic_labels = torch.randn(num_samples, PRED_LEN, INPUT_DIM)
        dataset = TensorDataset(synthetic_prices, synthetic_labels)

        # Wrap to match expected format
        class SyntheticDataset:
            def __init__(self, prices, labels):
                self.prices = prices
                self.labels = labels

            def __len__(self):
                return len(self.prices)

            def __getitem__(self, idx):
                return {"Price": self.prices[idx], "Labels": self.labels[idx]}

        dataset = SyntheticDataset(synthetic_prices, synthetic_labels)

    # Split into train and validation
    train_size = int(TRAIN_SPLIT * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    print(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # ========== Initialize Model ==========
    print("Initializing VAE model...")
    model = VAELightningModule(
        input_dim=INPUT_DIM,
        latent_dim=LATENT_DIM,
        d_model=D_MODEL,
        seq_len=SEQ_LEN,
        pred_len=PRED_LEN,
        encoder_type=ENCODER_TYPE,
        decoder_type=None,  # Use same as encoder
        n_layers=N_LAYERS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        kl_weight=KL_WEIGHT,
        kl_anneal_epochs=KL_ANNEAL_EPOCHS,
        reconstruction_loss=RECONSTRUCTION_LOSS,
        num_val_samples=8,
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ========== Setup Callbacks ==========
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/vae",
        filename="vae-{epoch:02d}-{val/loss:.4f}",
        monitor="val/loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )

    early_stopping_callback = EarlyStopping(
        monitor="val/loss", patience=15, mode="min", verbose=True
    )

    # ========== Setup Logger ==========
    logger = TensorBoardLogger(save_dir="logs", name="vae", version=None)

    # ========== Initialize Trainer ==========
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator=ACCELERATOR,
        devices=DEVICES,
        precision=PRECISION,
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
        log_every_n_steps=10,
        val_check_interval=1.0,
        gradient_clip_val=1.0,
    )

    # ========== Train ==========
    print("\nStarting training...")
    trainer.fit(model, train_loader, val_loader)

    print("\nTraining completed!")
    print(f"Best model saved at: {checkpoint_callback.best_model_path}")

    # ========== Generate Samples ==========
    print("\nGenerating samples from trained model...")
    model.eval()
    with torch.no_grad():
        samples = model.model.sample(num_samples=10, device=model.device)
        print(f"Generated samples shape: {samples.shape}")
        print(
            f"Sample statistics - Mean: {samples.mean():.4f}, Std: {samples.std():.4f}"
        )

    # ========== Evaluate Reconstruction ==========
    print("\nEvaluating reconstruction quality...")
    val_batch = next(iter(val_loader))
    val_batch = {k: v.to(model.device) for k, v in val_batch.items()}

    with torch.no_grad():
        # Get reconstruction
        reconstruction = model.model.reconstruct(val_batch["Price"], use_mean=True)
        target = val_batch["Price"][:, -PRED_LEN:, :]

        # Compute metrics
        mse = torch.nn.functional.mse_loss(reconstruction, target)
        mae = torch.nn.functional.l1_loss(reconstruction, target)

        print(f"Reconstruction MSE: {mse.item():.6f}")
        print(f"Reconstruction MAE: {mae.item():.6f}")

    # ========== Latent Space Analysis ==========
    print("\nAnalyzing latent space...")
    all_latents = []
    with torch.no_grad():
        for batch in val_loader:
            batch_dict = {k: v.to(model.device) for k, v in batch.items()}
            mean, _ = model.model.encode(batch_dict["Price"])
            all_latents.append(mean.cpu())

    all_latents = torch.cat(all_latents, dim=0)
    print(f"Latent space shape: {all_latents.shape}")
    print(
        f"Latent statistics - Mean: {all_latents.mean():.4f}, Std: {all_latents.std():.4f}"
    )
    print(f"Latent range - Min: {all_latents.min():.4f}, Max: {all_latents.max():.4f}")

    print("\n=== Training Script Completed ===")
    print(f"To view training logs, run: tensorboard --logdir {logger.log_dir}")


if __name__ == "__main__":
    main()
