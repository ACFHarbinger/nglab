# Variational Auto-Encoder (VAE) for Time Series Prediction

## Overview

This implementation provides a complete Variational Auto-Encoder (VAE) framework for time series prediction, fully integrated with NGLab's architecture. The VAE uses flexible backbone architectures (Transformer, Mamba, LSTM, GRU, xLSTM) and is designed for financial time series forecasting.

## Features

- **Flexible Backbones**: Support for multiple sequence models (Transformer, Mamba, LSTM, GRU, xLSTM)
- **Beta-VAE**: Controllable latent capacity via beta parameter
- **KL Annealing**: Gradual increase of KL weight for stable training
- **Multiple Reconstruction Losses**: MSE, L1, and Huber loss options
- **PyTorch Lightning Integration**: Scalable training with built-in logging and checkpointing
- **Sample Generation**: Generate new time series from learned latent distribution
- **Latent Space Analysis**: Tools for analyzing and visualizing the learned representations

## Architecture

The VAE consists of three main components:

### 1. Encoder
Maps input sequences to a latent distribution:
```
Input (batch, seq_len, input_dim)
  ↓
TimeSeriesBackbone (Transformer/Mamba/LSTM/etc.)
  ↓
Embedding (batch, d_model)
  ↓
[fc_mean, fc_log_var]
  ↓
[mean, log_var] (batch, latent_dim)
```

### 2. Reparameterization
Samples from the latent distribution using the reparameterization trick:
```
z = mean + std * ε, where ε ~ N(0, 1)
```

This allows backpropagation through the sampling operation.

### 3. Decoder
Maps latent samples back to output sequences:
```
Latent z (batch, latent_dim)
  ↓
Projection (batch, d_model)
  ↓
Expansion (batch, pred_len, d_model)
  ↓
TimeSeriesBackbone
  ↓
Reconstruction (batch, pred_len, input_dim)
```

## Loss Function

The VAE loss combines two terms:

### Total Loss
```
L = L_reconstruction + β * L_KL
```

### Reconstruction Loss
Measures how well the decoder reconstructs the input:
- **MSE**: Mean Squared Error (default)
- **L1**: Mean Absolute Error
- **Huber**: Smooth L1 loss (robust to outliers)

### KL Divergence
Regularizes the latent space to be close to N(0, 1):
```
L_KL = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
```

### Beta Parameter
Controls the trade-off between reconstruction and regularization:
- β = 1.0: Standard VAE
- β > 1.0: Beta-VAE (stronger regularization, more disentangled representations)
- β < 1.0: Weaker regularization (better reconstruction)

## Usage

### Basic Training

```python
from pipeline.lightning.vae_module import VAELightningModule
import pytorch_lightning as pl

# Initialize model
model = VAELightningModule(
    input_dim=5,          # OHLCV features
    latent_dim=32,        # Latent space dimension
    d_model=128,          # Hidden dimension
    seq_len=100,          # Input sequence length
    pred_len=20,          # Prediction horizon
    encoder_type='mamba', # Backbone type
    n_layers=3,           # Number of layers
    learning_rate=1e-3,
    kl_weight=1.0,        # Beta parameter
    kl_anneal_epochs=10   # KL annealing duration
)

# Train
trainer = pl.Trainer(max_epochs=100, accelerator='auto')
trainer.fit(model, train_loader, val_loader)
```

### Using the Example Script

```bash
cd python/examples
python train_vae.py
```

### Using Hydra Configuration

```bash
cd python/src
python main.py task=vae model=vae
```

## Configuration

### Model Configuration (`conf/model/vae.yaml`)

```yaml
input_dim: 5
latent_dim: 32
d_model: 128
seq_len: 100
pred_len: 20
encoder_type: mamba
n_layers: 3
learning_rate: 0.001
kl_weight: 1.0
kl_anneal_epochs: 10
reconstruction_loss: mse
```

### Task Configuration (`conf/task/vae.yaml`)

```yaml
task: vae
batch_size: 64
num_epochs: 100
accelerator: auto
early_stopping: true
patience: 15
```

## Advanced Features

### KL Annealing

KL annealing gradually increases the KL weight from 0 to the target β over a specified number of epochs. This prevents "posterior collapse" where the model ignores the latent space.

```python
model = VAELightningModule(
    kl_weight=1.0,         # Target beta value
    kl_anneal_epochs=20,   # Anneal over 20 epochs
    ...
)
```

The KL weight follows:
```
β(epoch) = target_β * min(1, epoch / anneal_epochs)
```

### Different Backbone Architectures

```python
# Mamba (State-Space Model)
model = VAELightningModule(encoder_type='mamba', ...)

# Transformer
model = VAELightningModule(encoder_type='transformer', n_heads=8, d_ff=512, ...)

# LSTM
model = VAELightningModule(encoder_type='lstm', ...)

# xLSTM (Extended LSTM)
model = VAELightningModule(encoder_type='xlstm', ...)

# Mixed backbones
model = VAELightningModule(
    encoder_type='mamba',
    decoder_type='transformer',  # Different decoder
    ...
)
```

### Generating Samples

```python
# Load trained model
model = VAELightningModule.load_from_checkpoint('path/to/checkpoint.ckpt')
model.eval()

# Generate samples from prior
samples = model.model.sample(num_samples=10, device='cuda')
# Shape: (10, pred_len, input_dim)
```

### Reconstruction

```python
# Deterministic reconstruction (using mean)
reconstruction = model.model.reconstruct(input_sequence, use_mean=True)

# Stochastic reconstruction (sampling)
reconstruction = model.model.reconstruct(input_sequence, use_mean=False)
```

### Latent Space Analysis

```python
# Encode data to latent space
mean, log_var = model.model.encode(input_sequence)

# Get latent representation
z = model.model.reparameterize(mean, log_var)

# Use latent for downstream tasks
# e.g., clustering, anomaly detection, etc.
```

## Use Cases

### 1. Time Series Forecasting
Train on historical data to predict future values:
```python
model = VAELightningModule(seq_len=100, pred_len=20, ...)
# Input: Last 100 time steps → Output: Next 20 time steps
```

### 2. Anomaly Detection
Use reconstruction error as anomaly score:
```python
reconstruction = model.model.reconstruct(normal_data, use_mean=True)
anomaly_score = torch.nn.functional.mse_loss(reconstruction, target, reduction='none').mean(dim=(1,2))
```

### 3. Feature Learning
Use latent representations for downstream tasks:
```python
latent_features, _ = model.model.encode(time_series)
# Use latent_features for classification, regression, etc.
```

### 4. Data Augmentation
Generate synthetic time series:
```python
synthetic_data = model.model.sample(num_samples=1000, device='cuda')
```

### 5. Denoising
Reconstruct clean signals from noisy inputs:
```python
clean_signal = model.model.reconstruct(noisy_signal, use_mean=True)
```

## Hyperparameter Tuning

### Latent Dimension
- **Small (8-16)**: Fast training, may lose information
- **Medium (32-64)**: Good balance (recommended)
- **Large (128+)**: More capacity, slower training

### Beta (KL Weight)
- **β = 0.1-0.5**: Prioritize reconstruction (for forecasting)
- **β = 1.0**: Standard VAE (balanced)
- **β = 2.0-10.0**: Prioritize disentanglement (for representation learning)

### KL Annealing
- **No annealing (0 epochs)**: Simple, but risk of posterior collapse
- **Short (5-10 epochs)**: Quick warmup
- **Long (20-50 epochs)**: More stable, recommended for complex models

### Reconstruction Loss
- **MSE**: Standard, sensitive to outliers
- **L1**: More robust to outliers
- **Huber**: Best of both worlds

## Monitoring Training

### TensorBoard Metrics

```bash
tensorboard --logdir logs/vae
```

Key metrics to monitor:
- `train/loss`, `val/loss`: Total loss
- `train/reconstruction_loss`, `val/reconstruction_loss`: Reconstruction quality
- `train/kl_loss`, `val/kl_loss`: KL divergence
- `train/latent_mean`, `train/latent_std`: Latent statistics
- `train/kl_weight`: Current beta value (for annealing)

### Good Training Signs
- Reconstruction loss decreases steadily
- KL loss stabilizes (not 0, not exploding)
- Latent mean ≈ 0, latent std ≈ 1
- Validation loss tracks training loss

### Warning Signs
- **Posterior Collapse**: KL loss → 0, latent std → 0
  - Solution: Increase kl_anneal_epochs, decrease kl_weight
- **Poor Reconstruction**: Reconstruction loss not decreasing
  - Solution: Decrease kl_weight, increase model capacity
- **Overfitting**: Val loss >> Train loss
  - Solution: Add dropout, use weight decay, get more data

## Testing

Run unit tests:
```bash
cd python
pytest tests/test_vae.py -v
```

Test coverage includes:
- Model architecture
- Forward/backward passes
- Loss computation
- KL annealing
- Sample generation
- Lightning module integration

## Performance Tips

1. **Mixed Precision Training**: Use `precision=16` for faster training
2. **Gradient Clipping**: Already enabled (1.0) to prevent exploding gradients
3. **Batch Size**: Larger batches (64-128) for stable KL estimates
4. **Learning Rate**: Start with 1e-3, reduce if unstable
5. **Model Size**: Use Mamba or LSTM for long sequences (>100 steps)

## References

1. Kingma & Welling (2013). "Auto-Encoding Variational Bayes" [[arXiv]](https://arxiv.org/abs/1312.6114)
2. Higgins et al. (2017). "β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework" [[PDF]](https://openreview.net/pdf?id=Sy2fzU9gl)
3. Sønderby et al. (2016). "How to Train Deep Variational Autoencoders and Probabilistic Ladder Networks" [[arXiv]](https://arxiv.org/abs/1602.02282)

## Troubleshooting

### Q: Model not learning / loss is NaN
**A**: Check your learning rate (try 1e-4), increase kl_anneal_epochs, verify input data is normalized

### Q: Posterior collapse (KL → 0)
**A**: Increase kl_anneal_epochs, set kl_weight < 1.0, reduce model capacity

### Q: Poor reconstruction quality
**A**: Decrease kl_weight, increase latent_dim, use more layers, try different backbone

### Q: Out of memory
**A**: Reduce batch_size, use smaller d_model, reduce n_layers, use precision=16

### Q: Slow training
**A**: Use Mamba or LSTM (faster than Transformer), reduce n_layers, use mixed precision

## File Structure

```
python/
├── src/
│   ├── models/
│   │   └── vae.py                      # VAE model implementation
│   ├── pipeline/
│   │   └── lightning/
│   │       └── vae_module.py           # Lightning training module
│   ├── conf/
│   │   ├── model/
│   │   │   └── vae.yaml               # Model configuration
│   │   └── task/
│   │       └── vae.yaml               # Task configuration
│   └── main.py                         # Main entry point (updated)
├── examples/
│   └── train_vae.py                    # Standalone training script
├── tests/
│   └── test_vae.py                     # Unit tests
└── docs/
    └── VAE_GUIDE.md                    # This guide
```

## Contributing

When extending the VAE implementation:
1. Add new features to `models/vae.py`
2. Update `pipeline/lightning/vae_module.py` for training integration
3. Add tests to `tests/test_vae.py`
4. Update this guide with examples
