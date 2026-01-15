"""
Unit tests for Variational Auto-Encoder implementation.

Tests cover:
1. Model architecture and forward pass
2. Encoding and decoding
3. Reparameterization trick
4. Loss computation
5. Sample generation
6. Lightning module integration
"""

import pytest
import torch
import torch.nn as nn
from typing import Dict

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.vae import TimeSeriesVAE, vae_loss
from pipeline.lightning.vae_module import VAELightningModule


class TestTimeSeriesVAE:
    """Tests for TimeSeriesVAE model."""

    @pytest.fixture
    def model_config(self):
        """Basic model configuration for testing."""
        return {
            'input_dim': 5,
            'latent_dim': 16,
            'd_model': 32,
            'seq_len': 20,
            'pred_len': 5,
            'encoder_type': 'lstm',
            'n_layers': 2,
            'dropout': 0.1
        }

    @pytest.fixture
    def model(self, model_config):
        """Create a VAE model for testing."""
        return TimeSeriesVAE(**model_config)

    @pytest.fixture
    def sample_batch(self, model_config):
        """Create a sample batch of data."""
        batch_size = 8
        return torch.randn(
            batch_size,
            model_config['seq_len'],
            model_config['input_dim']
        )

    def test_model_initialization(self, model, model_config):
        """Test that model initializes correctly."""
        assert model.input_dim == model_config['input_dim']
        assert model.latent_dim == model_config['latent_dim']
        assert model.d_model == model_config['d_model']
        assert model.seq_len == model_config['seq_len']
        assert model.pred_len == model_config['pred_len']

        # Check that submodules exist
        assert hasattr(model, 'encoder')
        assert hasattr(model, 'decoder')
        assert hasattr(model, 'fc_mean')
        assert hasattr(model, 'fc_log_var')

    def test_forward_pass(self, model, sample_batch):
        """Test forward pass through the model."""
        output = model(sample_batch)

        # Check output structure
        assert isinstance(output, dict)
        assert 'reconstruction' in output
        assert 'mean' in output
        assert 'log_var' in output
        assert 'z' in output

        # Check output shapes
        batch_size = sample_batch.size(0)
        assert output['reconstruction'].shape == (batch_size, model.pred_len, model.input_dim)
        assert output['mean'].shape == (batch_size, model.latent_dim)
        assert output['log_var'].shape == (batch_size, model.latent_dim)
        assert output['z'].shape == (batch_size, model.latent_dim)

    def test_encode(self, model, sample_batch):
        """Test encoding to latent space."""
        mean, log_var = model.encode(sample_batch)

        batch_size = sample_batch.size(0)
        assert mean.shape == (batch_size, model.latent_dim)
        assert log_var.shape == (batch_size, model.latent_dim)

        # Check that outputs are finite
        assert torch.isfinite(mean).all()
        assert torch.isfinite(log_var).all()

    def test_reparameterization(self, model):
        """Test reparameterization trick."""
        batch_size = 8
        mean = torch.zeros(batch_size, model.latent_dim)
        log_var = torch.zeros(batch_size, model.latent_dim)

        # Sample multiple times
        samples = [model.reparameterize(mean, log_var) for _ in range(100)]
        samples = torch.stack(samples)

        # Check that samples are different (stochastic)
        assert not torch.allclose(samples[0], samples[1])

        # Check that samples have approximately correct statistics
        # Mean should be close to 0, std close to 1
        sample_mean = samples.mean(dim=0).abs().mean()
        sample_std = samples.std(dim=0).mean()

        assert sample_mean < 0.2  # Should be close to 0
        assert 0.8 < sample_std < 1.2  # Should be close to 1

    def test_decode(self, model):
        """Test decoding from latent space."""
        batch_size = 8
        z = torch.randn(batch_size, model.latent_dim)

        reconstruction = model.decode(z)

        assert reconstruction.shape == (batch_size, model.pred_len, model.input_dim)
        assert torch.isfinite(reconstruction).all()

    def test_reconstruct(self, model, sample_batch):
        """Test reconstruction with and without mean."""
        # Reconstruction with sampling
        recon_sample = model.reconstruct(sample_batch, use_mean=False)
        assert recon_sample.shape == (sample_batch.size(0), model.pred_len, model.input_dim)

        # Reconstruction with mean (deterministic)
        recon_mean = model.reconstruct(sample_batch, use_mean=True)
        assert recon_mean.shape == (sample_batch.size(0), model.pred_len, model.input_dim)

        # Deterministic reconstruction should be identical
        recon_mean2 = model.reconstruct(sample_batch, use_mean=True)
        assert torch.allclose(recon_mean, recon_mean2)

    def test_sample_generation(self, model):
        """Test generating samples from prior."""
        num_samples = 10
        device = torch.device('cpu')

        samples = model.sample(num_samples, device)

        assert samples.shape == (num_samples, model.pred_len, model.input_dim)
        assert torch.isfinite(samples).all()

    @pytest.mark.parametrize('encoder_type', ['lstm', 'gru', 'mamba'])
    def test_different_backbones(self, encoder_type, model_config):
        """Test VAE with different backbone architectures."""
        config = model_config.copy()
        config['encoder_type'] = encoder_type

        model = TimeSeriesVAE(**config)
        batch = torch.randn(4, config['seq_len'], config['input_dim'])

        output = model(batch)

        assert output['reconstruction'].shape == (4, config['pred_len'], config['input_dim'])

    def test_gradient_flow(self, model, sample_batch):
        """Test that gradients flow through the model."""
        model.train()
        output = model(sample_batch)

        # Compute dummy loss
        loss = output['reconstruction'].mean() + output['mean'].mean() + output['log_var'].mean()
        loss.backward()

        # Check that gradients exist and are non-zero
        has_gradient = False
        for param in model.parameters():
            if param.grad is not None:
                if param.grad.abs().sum() > 0:
                    has_gradient = True
                    break

        assert has_gradient, "No gradients computed"


class TestVAELoss:
    """Tests for VAE loss function."""

    @pytest.fixture
    def loss_inputs(self):
        """Create sample inputs for loss computation."""
        batch_size, pred_len, input_dim = 8, 5, 3
        latent_dim = 16

        return {
            'reconstruction': torch.randn(batch_size, pred_len, input_dim),
            'target': torch.randn(batch_size, pred_len, input_dim),
            'mean': torch.randn(batch_size, latent_dim),
            'log_var': torch.randn(batch_size, latent_dim)
        }

    def test_loss_computation(self, loss_inputs):
        """Test basic loss computation."""
        total_loss, loss_dict = vae_loss(**loss_inputs)

        # Check that loss is computed
        assert isinstance(total_loss, torch.Tensor)
        assert total_loss.ndim == 0  # Scalar

        # Check loss components
        assert 'total_loss' in loss_dict
        assert 'reconstruction_loss' in loss_dict
        assert 'kl_loss' in loss_dict
        assert 'kl_weighted' in loss_dict

        # Check that all losses are finite and positive
        for key, value in loss_dict.items():
            assert torch.isfinite(value)
            assert value >= 0

    @pytest.mark.parametrize('kl_weight', [0.0, 0.5, 1.0, 2.0])
    def test_kl_weight(self, loss_inputs, kl_weight):
        """Test KL weight parameter (beta-VAE)."""
        total_loss, loss_dict = vae_loss(**loss_inputs, kl_weight=kl_weight)

        expected_weighted = loss_dict['kl_loss'] * kl_weight
        assert torch.isclose(loss_dict['kl_weighted'], expected_weighted)

        expected_total = loss_dict['reconstruction_loss'] + expected_weighted
        assert torch.isclose(total_loss, expected_total)

    @pytest.mark.parametrize('loss_type', ['mse', 'l1', 'huber'])
    def test_reconstruction_loss_types(self, loss_inputs, loss_type):
        """Test different reconstruction loss types."""
        total_loss, loss_dict = vae_loss(
            **loss_inputs,
            reconstruction_loss=loss_type
        )

        assert torch.isfinite(total_loss)
        assert torch.isfinite(loss_dict['reconstruction_loss'])

    def test_kl_divergence_properties(self, loss_inputs):
        """Test KL divergence properties."""
        # KL should be 0 when distribution is N(0, 1)
        loss_inputs['mean'] = torch.zeros_like(loss_inputs['mean'])
        loss_inputs['log_var'] = torch.zeros_like(loss_inputs['log_var'])

        _, loss_dict = vae_loss(**loss_inputs)

        # KL should be very close to 0
        assert loss_dict['kl_loss'] < 1e-5


class TestVAELightningModule:
    """Tests for VAE Lightning module."""

    @pytest.fixture
    def module_config(self):
        """Configuration for Lightning module."""
        return {
            'input_dim': 5,
            'latent_dim': 16,
            'd_model': 32,
            'seq_len': 20,
            'pred_len': 5,
            'encoder_type': 'lstm',
            'n_layers': 2,
            'learning_rate': 1e-3,
            'kl_weight': 1.0,
            'kl_anneal_epochs': 5
        }

    @pytest.fixture
    def lightning_module(self, module_config):
        """Create Lightning module for testing."""
        return VAELightningModule(**module_config)

    @pytest.fixture
    def sample_batch_dict(self, module_config):
        """Create sample batch in expected format."""
        batch_size = 8
        return {
            'Price': torch.randn(batch_size, module_config['seq_len'], module_config['input_dim'])
        }

    def test_module_initialization(self, lightning_module, module_config):
        """Test Lightning module initialization."""
        assert lightning_module.learning_rate == module_config['learning_rate']
        assert lightning_module.kl_weight == module_config['kl_weight']
        assert hasattr(lightning_module, 'model')
        assert isinstance(lightning_module.model, TimeSeriesVAE)

    def test_kl_annealing(self, lightning_module):
        """Test KL weight annealing."""
        # At epoch 0, KL weight should be 0
        lightning_module.current_epoch = 0
        assert lightning_module.get_current_kl_weight() == 0.0

        # At middle of annealing
        lightning_module.current_epoch = 2
        kl_anneal_epochs = lightning_module.kl_anneal_epochs
        expected = lightning_module.kl_weight * (2 / kl_anneal_epochs)
        assert lightning_module.get_current_kl_weight() == pytest.approx(expected)

        # After annealing is done
        lightning_module.current_epoch = 10
        assert lightning_module.get_current_kl_weight() == lightning_module.kl_weight

    def test_training_step(self, lightning_module, sample_batch_dict):
        """Test training step."""
        loss = lightning_module.training_step(sample_batch_dict, 0)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar
        assert torch.isfinite(loss)
        assert loss >= 0

    def test_validation_step(self, lightning_module, sample_batch_dict):
        """Test validation step."""
        loss = lightning_module.validation_step(sample_batch_dict, 0)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert torch.isfinite(loss)
        assert loss >= 0

    def test_predict_step(self, lightning_module, sample_batch_dict):
        """Test prediction step."""
        predictions = lightning_module.predict_step(sample_batch_dict, 0)

        assert isinstance(predictions, dict)
        assert 'reconstruction' in predictions
        assert 'latent_mean' in predictions
        assert 'latent_log_var' in predictions
        assert 'input' in predictions

        batch_size = sample_batch_dict['Price'].size(0)
        assert predictions['reconstruction'].shape[0] == batch_size
        assert predictions['latent_mean'].shape[0] == batch_size

    def test_configure_optimizers(self, lightning_module):
        """Test optimizer configuration."""
        # Need to set trainer for this test
        import pytorch_lightning as pl
        trainer = pl.Trainer(max_epochs=10, fast_dev_run=True)
        lightning_module.trainer = trainer

        optimizer_config = lightning_module.configure_optimizers()

        assert 'optimizer' in optimizer_config
        assert 'lr_scheduler' in optimizer_config

        optimizer = optimizer_config['optimizer']
        assert isinstance(optimizer, torch.optim.AdamW)


class TestVAEIntegration:
    """Integration tests for complete VAE pipeline."""

    def test_end_to_end_training(self):
        """Test complete training pipeline."""
        # Create small model
        model = VAELightningModule(
            input_dim=3,
            latent_dim=8,
            d_model=16,
            seq_len=10,
            pred_len=3,
            encoder_type='lstm',
            n_layers=1,
            learning_rate=1e-3,
            kl_anneal_epochs=2
        )

        # Create synthetic data
        from torch.utils.data import DataLoader, TensorDataset

        num_samples = 32
        prices = torch.randn(num_samples, 10, 3)
        labels = torch.randn(num_samples, 3, 3)

        class DictDataset:
            def __init__(self, prices, labels):
                self.prices = prices
                self.labels = labels

            def __len__(self):
                return len(self.prices)

            def __getitem__(self, idx):
                return {'Price': self.prices[idx], 'Labels': self.labels[idx]}

        dataset = DictDataset(prices, labels)
        dataloader = DataLoader(dataset, batch_size=8)

        # Train for a few steps
        import pytorch_lightning as pl
        trainer = pl.Trainer(
            max_epochs=2,
            accelerator='cpu',
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False
        )

        trainer.fit(model, dataloader, dataloader)

        # Test prediction
        predictions = trainer.predict(model, dataloader)
        assert len(predictions) > 0

    def test_sample_generation_quality(self):
        """Test that generated samples have reasonable properties."""
        model = TimeSeriesVAE(
            input_dim=3,
            latent_dim=8,
            d_model=16,
            seq_len=10,
            pred_len=5,
            encoder_type='lstm',
            n_layers=1
        )

        # Generate samples
        samples = model.sample(num_samples=100, device=torch.device('cpu'))

        # Check that samples have reasonable statistics
        assert torch.isfinite(samples).all()
        assert samples.std() > 0  # Not all zeros
        assert -10 < samples.mean() < 10  # Reasonable range


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
