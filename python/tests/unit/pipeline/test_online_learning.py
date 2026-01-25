import numpy as np
import torch

from python.src.pipeline.online_learning.drift import MovingAverageDrift, PageHinkley
from python.src.pipeline.online_learning.normalization import OnlineNormalizer


def test_page_hinkley_drift():
    ph = PageHinkley(min_instances=10, delta=0.1, threshold=10.0)

    # Stable phase
    for _ in range(20):
        drift = ph.update(100.0 + np.random.normal(0, 0.1))
        assert not drift

    # Drift phase
    drift_detected = False
    for _i in range(20):
        # Abrupt jump
        val = 110.0 + np.random.normal(0, 0.1)
        if ph.update(val):
            drift_detected = True
            break

    assert drift_detected

    # Check reset
    ph.reset()
    assert ph.sample_count == 0
    assert not ph.in_drift


def test_moving_average_drift():
    ma_drift = MovingAverageDrift(short_window=5, long_window=10, threshold=2.0)

    # Stable
    for _ in range(15):
        assert not ma_drift.update(100.0)

    # Drift
    # Sudden jump to 120.0
    # Short MA will rise faster than Long MA
    for _ in range(5):
        if ma_drift.update(120.0):
            break

    # Depending on window overlap, it might trigger.
    # Short MA (5) of 120 vs Long MA (5*100 + 5*120)/10 = 110
    # Diff = 10. Std dev roughly 10 (range 100-120). Z-score ~1.
    # Might need more drift or tighter threshold.
    # Let's use clean jump.

    ma_drift = MovingAverageDrift(short_window=5, long_window=20, threshold=1.0)
    # Fill with 100
    for _ in range(20):
        ma_drift.update(100.0)

    # Jump to 110
    drift = False
    for _ in range(5):
        if ma_drift.update(110.0):
            drift = True
            break

    # Short MA becomes 110. Long MA is (15*100 + 5*110)/20 = 102.5.
    # Diff = 7.5. Std dev of mix of 100 and 110 is ~3-4.
    # Z-score > 1.0 likely.
    assert drift


def test_online_normalizer_welford():
    norm = OnlineNormalizer(num_features=2, momentum=None, affine=False)

    # Data: feature 1 mean=10, feature 2 mean=20
    data = torch.tensor([[10.0, 20.0], [12.0, 22.0], [8.0, 18.0]])

    # Manual calculation
    # Mean: [10, 20]
    # Var: 1/3 * ((0)^2 + (2)^2 + (-2)^2) = 1/3 * 8 = 2.666...
    # Unbiased Var (torch default for var() is unbiased): 1/2 * 8 = 4
    # Our normalizer uses biased var (population var) logic for 'running_var' typically,
    # but let's check implementation.
    # Logic: m2_total / total_count -> This is biased variance (population).

    expected_mean = torch.tensor([10.0, 20.0])
    expected_var = torch.tensor([8.0 / 3.0, 8.0 / 3.0])

    # Update
    norm.update(data)

    assert torch.allclose(norm.running_mean, expected_mean, atol=0.1)
    assert torch.allclose(norm.running_var, expected_var, atol=0.1)

    # Forward pass
    out = norm(data, update_stats=False)  # Should normalize using stats
    # Expected out[0] = (10-10)/std = 0
    assert torch.allclose(out[0], torch.tensor([0.0, 0.0]), atol=1e-5)


def test_online_normalizer_momentum():
    norm = OnlineNormalizer(num_features=1, momentum=0.1, affine=False)

    # Initial state: mean=0, var=1

    # Batch 1: mean=10, var=0
    x = torch.tensor([[10.0]])
    norm.update(x)

    # New running mean = 0.9 * 0 + 0.1 * 10 = 1.0
    assert torch.allclose(norm.running_mean, torch.tensor([1.0]))

    # Update again
    norm.update(x)
    # New running mean = 0.9 * 1.0 + 0.1 * 10 = 0.9 + 1.0 = 1.9
    assert torch.allclose(norm.running_mean, torch.tensor([1.9]))
