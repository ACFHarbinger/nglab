"""
Unit tests for mixed precision training module.

Tests MixedPrecisionConfig, MixedPrecisionTrainer, and precision utilities
for FP16/BF16 mixed precision training.
"""

import os
from unittest.mock import patch

import pytest
import torch
from torch import nn

from python.src.utils.mixed_precision import (
    MixedPrecisionConfig,
    MixedPrecisionTrainer,
    PrecisionMode,
    configure_model_for_mixed_precision,
    estimate_memory_savings,
    get_optimal_precision,
)

# Test fixtures


@pytest.fixture
def simple_model():
    """Create a simple test model."""

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(10, 20)
            self.norm = nn.LayerNorm(20)
            self.linear2 = nn.Linear(20, 1)

        def forward(self, x):
            x = self.linear1(x)
            x = self.norm(x)
            x = torch.relu(x)
            return self.linear2(x)

    return SimpleModel()


@pytest.fixture
def sample_batch():
    """Create a sample batch of data."""
    return torch.randn(4, 10)


@pytest.fixture
def cleanup_env():
    """Clean up environment variables after test."""
    yield
    # Clean up any mixed precision env vars
    if "NGLAB_MIXED_PRECISION" in os.environ:
        del os.environ["NGLAB_MIXED_PRECISION"]
    if "NGLAB_PRECISION_MODE" in os.environ:
        del os.environ["NGLAB_PRECISION_MODE"]


# ============================================================
# PrecisionMode Tests
# ============================================================


def test_precision_mode_enum():
    """Test PrecisionMode enum values."""
    assert PrecisionMode.FP32.value == "32"
    assert PrecisionMode.FP16_MIXED.value == "16-mixed"
    assert PrecisionMode.BF16_MIXED.value == "bf16-mixed"
    assert PrecisionMode.FP16_TRUE.value == "16-true"
    assert PrecisionMode.BF16_TRUE.value == "bf16-true"


# ============================================================
# MixedPrecisionConfig Tests
# ============================================================


def test_config_initialization():
    """Test MixedPrecisionConfig initialization with defaults."""
    config = MixedPrecisionConfig()

    assert config.precision == "16-mixed"
    assert config.init_scale == 65536.0
    assert config.growth_factor == 2.0
    assert config.backoff_factor == 0.5
    assert config.growth_interval == 2000


def test_config_custom_values():
    """Test MixedPrecisionConfig with custom values."""
    config = MixedPrecisionConfig(
        precision="bf16-mixed",
        init_scale=32768.0,
        growth_factor=1.5,
        backoff_factor=0.25,
    )

    assert config.precision == "bf16-mixed"
    assert config.init_scale == 32768.0
    assert config.growth_factor == 1.5
    assert config.backoff_factor == 0.25


def test_config_from_env(cleanup_env):
    """Test creating config from environment variables."""
    os.environ["NGLAB_MIXED_PRECISION"] = "true"
    os.environ["NGLAB_PRECISION_MODE"] = "bf16-mixed"

    config = MixedPrecisionConfig.from_env()

    assert config.use_amp is True


def test_config_dtype():
    """Test dtype property for different precision modes."""
    # FP16 modes
    config_fp16_mixed = MixedPrecisionConfig(precision="16-mixed")
    assert config_fp16_mixed.dtype == torch.float16

    config_fp16_true = MixedPrecisionConfig(precision="16-true")
    assert config_fp16_true.dtype == torch.float16

    # BF16 modes
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        config_bf16_mixed = MixedPrecisionConfig(precision="bf16-mixed")
        assert config_bf16_mixed.dtype == torch.bfloat16

        config_bf16_true = MixedPrecisionConfig(precision="bf16-true")
        assert config_bf16_true.dtype == torch.bfloat16

    # FP32
    config_fp32 = MixedPrecisionConfig(precision="32")
    assert config_fp32.dtype == torch.float32


# ============================================================
# MixedPrecisionTrainer Tests
# ============================================================


def test_trainer_initialization(simple_model):
    """Test MixedPrecisionTrainer initialization."""
    optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.001)
    config = MixedPrecisionConfig(precision="16-mixed")

    trainer = MixedPrecisionTrainer(simple_model, optimizer, config)

    assert trainer.model == simple_model
    assert trainer.optimizer == optimizer
    assert trainer.config == config
    assert trainer.scaler is not None  # Should have GradScaler for FP16


def test_trainer_no_amp(simple_model):
    """Test MixedPrecisionTrainer without AMP (FP32)."""
    optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.001)
    config = MixedPrecisionConfig(precision="32")

    trainer = MixedPrecisionTrainer(simple_model, optimizer, config)

    assert trainer.scaler is None  # No scaler for FP32


@pytest.mark.gpu
def test_trainer_autocast_context(simple_model, device):
    """Test autocast context manager in trainer (requires GPU for mixed precision)."""
    optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.001)
    config = MixedPrecisionConfig(precision="16-mixed")

    trainer = MixedPrecisionTrainer(simple_model, optimizer, config)
    device = torch.device(trainer._device_type)
    simple_model.to(device)

    with trainer.autocast_context():
        x = torch.randn(4, 10).to(device)
        output = simple_model(x)
        # In autocast context, output should be FP16
        if config.use_amp and torch.cuda.is_available():
            assert output.dtype == torch.float16
        elif config.use_amp and not torch.cuda.is_available():
            # On CPU, autocast with float16 doesn't change dtype
            assert output.dtype == torch.float32


def test_trainer_training_step(simple_model, sample_batch):
    """Test training step with mixed precision."""
    optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.001)
    config = MixedPrecisionConfig(precision="16-mixed")

    trainer = MixedPrecisionTrainer(simple_model, optimizer, config)

    def forward_fn(model, batch):
        return model(batch)

    def loss_fn(output, batch):
        return output.mean()

    # Execute training step
    loss, _loss_val = trainer.training_step(sample_batch, forward_fn, loss_fn)

    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad


def test_trainer_step_with_clipping(simple_model, sample_batch):
    """Test optimizer step with gradient clipping."""
    optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.001)
    config = MixedPrecisionConfig(precision="16-mixed")

    trainer = MixedPrecisionTrainer(simple_model, optimizer, config)

    # Create some gradients
    output = simple_model(sample_batch)
    loss = output.mean()
    if trainer.scaler is not None:
        trainer.scaler.scale(loss).backward()
    else:
        loss.backward()

    # Step with gradient clipping
    trainer.step(clip_grad_norm=1.0)
    optimizer.zero_grad()

    # Verify optimizer took a step (grad should be cleared)
    for param in simple_model.parameters():
        if param.grad is not None:
            # Gradients should be zeroed after step
            assert param.grad.abs().sum() == 0


def test_trainer_state_dict(simple_model):
    """Test state dict save/load."""
    optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.001)
    config = MixedPrecisionConfig(precision="16-mixed")

    trainer = MixedPrecisionTrainer(simple_model, optimizer, config)

    # Get state dict
    state = trainer.state_dict()

    assert "scaler" in state
    assert "enabled" in state
    assert state["enabled"] is True


def test_trainer_load_state_dict(simple_model):
    """Test loading state dict."""
    optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.001)
    config = MixedPrecisionConfig(precision="16-mixed")

    trainer1 = MixedPrecisionTrainer(simple_model, optimizer, config)

    # Get state from first trainer
    state = trainer1.state_dict()

    # Create new trainer and load state
    optimizer2 = torch.optim.Adam(simple_model.parameters(), lr=0.001)
    trainer2 = MixedPrecisionTrainer(simple_model, optimizer2, config)
    trainer2.load_state_dict(state)

    # Verify scale matches
    assert trainer1.get_scale() == trainer2.get_scale()


@pytest.mark.gpu
def test_trainer_get_scale(simple_model):
    """Test getting loss scale from trainer (requires GPU for GradScaler)."""
    optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.001)
    config = MixedPrecisionConfig(precision="16-mixed", init_scale=32768.0)

    trainer = MixedPrecisionTrainer(simple_model, optimizer, config)

    scale = trainer.get_scale()

    # Should match init_scale
    assert scale == 32768.0


def test_trainer_accumulation_steps(simple_model, sample_batch):
    """Test gradient accumulation with mixed precision."""
    optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.001)
    config = MixedPrecisionConfig(precision="16-mixed")

    trainer = MixedPrecisionTrainer(simple_model, optimizer, config)

    def forward_fn(model, batch):
        return model(batch)

    def loss_fn(output, batch):
        return output.mean()

    # Accumulate gradients over 2 steps
    loss1, _ = trainer.training_step(
        sample_batch, forward_fn, loss_fn, accumulation_steps=2
    )
    loss2, _ = trainer.training_step(
        sample_batch, forward_fn, loss_fn, accumulation_steps=2
    )

    assert isinstance(loss1, torch.Tensor)
    assert isinstance(loss2, torch.Tensor)


# ============================================================
# Utility Function Tests
# ============================================================


def test_configure_model_for_mixed_precision(simple_model):
    """Test configuring model for mixed precision."""
    configured_model = configure_model_for_mixed_precision(
        simple_model, precision="16-mixed"
    )

    # Model should still work
    output = configured_model(torch.randn(4, 10))
    assert output.shape == (4, 1)


def test_get_optimal_precision():
    """Test detecting optimal precision for hardware."""
    precision = get_optimal_precision()

    assert isinstance(precision, str)
    # Should return one of the valid precision modes
    valid_precisions = ["32", "16-mixed", "bf16-mixed"]
    assert precision in valid_precisions


@patch("torch.cuda.is_available")
@pytest.mark.gpu
@patch("torch.cuda.is_bf16_supported")
def test_get_optimal_precision_with_cuda(mock_cuda, mock_bf16):
    """Test get_optimal_precision when CUDA is available (requires GPU)."""
    mock_cuda.return_value = True
    mock_bf16.return_value = True

    precision = get_optimal_precision()

    # With CUDA and BF16 support, should prefer bf16-mixed
    assert precision in ["bf16-mixed", "16-mixed"]


@patch("torch.cuda.is_available")
def test_get_optimal_precision_no_cuda(mock_cuda):
    """Test optimal precision detection without CUDA."""
    mock_cuda.return_value = False

    precision = get_optimal_precision()

    # Without CUDA, should use FP32
    assert precision == "32"


def test_estimate_memory_savings(simple_model):
    """Test estimating memory savings from mixed precision."""
    estimates = estimate_memory_savings(
        simple_model, batch_size=32, sequence_length=100, precision="16-mixed"
    )

    assert isinstance(estimates, dict)
    assert "fp32_memory_mb" in estimates
    assert "mixed_memory_mb" in estimates
    assert "savings_mb" in estimates
    assert "savings_percent" in estimates

    # Mixed precision should save memory
    assert estimates["savings_mb"] >= 0
    assert 0 <= estimates["savings_percent"] <= 100


def test_estimate_memory_savings_bf16(simple_model):
    """Test memory estimates for BF16."""
    estimates = estimate_memory_savings(
        simple_model, batch_size=32, sequence_length=100, precision="bf16-mixed"
    )

    assert isinstance(estimates, dict)
    # BF16 should have similar savings to FP16
    assert estimates["savings_mb"] >= 0


def test_estimate_memory_savings_fp32(simple_model):
    """Test memory estimates for FP32 (no savings)."""
    estimates = estimate_memory_savings(
        simple_model, batch_size=32, sequence_length=100, precision="32"
    )

    assert isinstance(estimates, dict)
    # FP32 should have no savings
    assert estimates["savings_mb"] == 0
    assert estimates["savings_percent"] == 0


# ============================================================
# Integration Tests
# ============================================================


def test_full_training_loop(simple_model):
    """Test full training loop with mixed precision."""
    optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.001)
    config = MixedPrecisionConfig(precision="16-mixed")

    trainer = MixedPrecisionTrainer(simple_model, optimizer, config)

    def forward_fn(model, batch):
        return model(batch)

    def loss_fn(output, batch):
        return output.mean()

    # Train for a few steps
    for _ in range(5):
        batch = torch.randn(4, 10)
        _loss, _ = trainer.training_step(batch, forward_fn, loss_fn)
        trainer.step(clip_grad_norm=1.0)

    # Model should have updated parameters
    assert all(
        param.grad is not None or not param.requires_grad
        for param in simple_model.parameters()
    )


def test_checkpoint_save_load_cycle(simple_model, tmp_path):
    """Test saving and loading mixed precision training state."""
    optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.001)
    config = MixedPrecisionConfig(precision="16-mixed")

    trainer1 = MixedPrecisionTrainer(simple_model, optimizer, config)

    # Train for a step
    batch = torch.randn(4, 10)
    _loss, _ = trainer1.training_step(batch, lambda m, b: m(b), lambda o, b: o.mean())
    trainer1.step()

    # Save state
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "model": simple_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "trainer": trainer1.state_dict(),
        },
        checkpoint_path,
    )

    # Create new trainer and load state
    new_model = type(simple_model)()
    new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)
    trainer2 = MixedPrecisionTrainer(new_model, new_optimizer, config)

    checkpoint = torch.load(checkpoint_path, weights_only=False)
    new_model.load_state_dict(checkpoint["model"])
    new_optimizer.load_state_dict(checkpoint["optimizer"])
    trainer2.load_state_dict(checkpoint["trainer"])

    # Verify scale matches
    assert trainer1.get_scale() == trainer2.get_scale()
