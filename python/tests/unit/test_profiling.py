from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

from python.src.utils.profiling.cuda_profiler import (
    CUDAProfiler,
    ProfilerConfig,
    get_gpu_memory_stats,
    profile_model_forward,
    profile_training_step,
)


@pytest.fixture
def dummy_model():
    return nn.Linear(10, 2)


@pytest.fixture
def dummy_optimizer(dummy_model):
    return torch.optim.Adam(dummy_model.parameters())


def test_profiler_config():
    config = ProfilerConfig(output_dir="test_output", profile_memory=False)
    assert config.output_dir == "test_output"
    assert config.profile_memory is False


@patch("python.src.utils.profiling.cuda_profiler.torch_profile")
def test_cuda_profiler_basic(mock_torch_profile, tmp_path):
    config = ProfilerConfig(
        output_dir=str(tmp_path / "profiler"), export_chrome_trace=False
    )
    profiler = CUDAProfiler(config)

    # Mock profiler object
    mock_prof_obj = MagicMock()
    mock_torch_profile.return_value.__enter__.return_value = mock_prof_obj

    with profiler.profile() as p:
        p.step()
        assert p._step_count == 1

    assert profiler.get_results() is not None


@patch("torch.cuda.is_available", return_value=True)
@patch("torch.cuda.synchronize")
@patch("torch.cuda.memory_allocated", return_value=1024 * 1024 * 100)  # 100MB
@patch("torch.cuda.memory_reserved", return_value=1024 * 1024 * 200)  # 200MB
@patch("torch.cuda.max_memory_allocated", return_value=1024 * 1024 * 150)
@patch("torch.cuda.max_memory_reserved", return_value=1024 * 1024 * 250)
@patch("torch.cuda.get_device_properties")
def test_get_gpu_memory_stats(mock_props, *args):
    mock_props.return_value.total_memory = 1024 * 1024 * 1000  # 1GB

    stats = get_gpu_memory_stats(device=0)

    assert stats is not None
    assert stats.allocated_mb == 100.0
    assert stats.total_mb == 1000.0
    assert stats.utilization_percent == 10.0


@patch("torch.cuda.is_available", return_value=False)
def test_get_gpu_memory_stats_no_cuda(mock_cuda):
    assert get_gpu_memory_stats() is None


@patch("torch.cuda.is_available", return_value=False)
@patch("time.perf_counter", side_effect=[0.0, 0.001, 0.0, 0.001])  # Mock timing for CPU
def test_profile_model_forward(mock_time, mock_cuda, dummy_model):
    input_tensor = torch.randn(1, 10)
    results = profile_model_forward(
        dummy_model, input_tensor, num_iterations=1, warmup_iterations=1
    )

    assert "mean_ms" in results
    assert results["mean_ms"] == 1.0  # 0.001 * 1000


@patch("torch.cuda.is_available", return_value=False)
@patch(
    "time.perf_counter", side_effect=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
)  # Alternating start/end
def test_profile_training_step(mock_time, mock_cuda, dummy_model, dummy_optimizer):
    input_tensor = torch.randn(1, 10)
    target_tensor = torch.randn(1, 2)
    loss_fn = nn.MSELoss()

    results = profile_training_step(
        dummy_model,
        dummy_optimizer,
        loss_fn,
        input_tensor,
        target_tensor,
        num_iterations=2,
        warmup_iterations=1,
    )

    assert "mean_ms" in results
    assert "avg_loss" in results
    assert results["peak_memory_mb"] == 0.0  # CPU
