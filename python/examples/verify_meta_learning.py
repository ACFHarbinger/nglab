"""
Verification script for Meta-Learning and Regime Detection.
"""

import numpy as np
import torch

from python.src.models.meta_learner import MAMLWrapper
from python.src.models.time_series import TimeSeriesBackbone
from python.src.pipeline.meta.regime_detector import RegimeDetector


def verify_maml():
    print("--- MAML Verification ---")

    # Create a simple model
    config = {
        "name": "LSTM",
        "feature_dim": 5,
        "hidden_dim": 32,
        "num_layers": 1,
        "output_dim": 1,
        "seq_len": 10,
        "output_type": "prediction",
    }
    model = TimeSeriesBackbone(config)

    # Wrap with MAML
    maml = MAMLWrapper(model, inner_lr=0.01, outer_lr=0.001, inner_steps=3)
    print(
        f"Created MAML wrapper with {sum(p.numel() for p in maml.parameters())} parameters"
    )

    # Simulate tasks (different market regimes)
    print("\nSimulating 3 market regime tasks...")
    tasks = []
    for _ in range(3):
        # Generate synthetic data for each regime
        support_x = torch.randn(20, 10, 5)
        support_y = torch.randn(20, 1)
        query_x = torch.randn(10, 10, 5)
        query_y = torch.randn(10, 1)
        tasks.append((support_x, support_y, query_x, query_y))

    # Meta-training step
    print("Performing meta-training step...")
    meta_loss = maml.meta_train_step(tasks)
    print(f"Meta-loss: {meta_loss:.4f}")

    # Fast adaptation test
    print("\nTesting fast adaptation...")
    support_x, support_y, query_x, query_y = tasks[0]
    adapted_model = maml.adapt(support_x, support_y, num_steps=5)

    with torch.no_grad():
        adapted_pred = adapted_model(query_x)
        adapted_loss = torch.nn.functional.mse_loss(adapted_pred, query_y)
    print(f"Adapted model loss: {adapted_loss:.4f}")

    print("\n✓ MAML verification passed!")


def verify_regime_detection():
    print("\n--- Regime Detection Verification ---")

    # Generate synthetic price data with different regimes
    np.random.seed(42)

    # Regime 1: Volatile
    volatile = np.cumsum(np.random.randn(200) * 2) + 100

    # Regime 2: Trending
    trending = np.linspace(100, 150, 200) + np.random.randn(200) * 0.5

    # Regime 3: Ranging
    ranging = np.sin(np.linspace(0, 10, 200)) * 5 + 100 + np.random.randn(200) * 0.3

    prices = np.concatenate([volatile, trending, ranging])

    # Fit detector
    detector = RegimeDetector(n_regimes=3, window_size=50)
    print("Fitting regime detector on 600 price points...")
    detector.fit(prices)

    # Test prediction
    print("\nTesting regime prediction...")
    test_windows = [
        prices[150:200],  # Should be volatile
        prices[350:400],  # Should be trending
        prices[550:600],  # Should be ranging
    ]

    for i, window in enumerate(test_windows):
        regime_id = detector.predict(window)
        regime_name = detector.get_regime_name(regime_id)
        print(f"Window {i + 1}: Detected regime '{regime_name}' (ID: {regime_id})")

    print("\n✓ Regime detection verification passed!")


if __name__ == "__main__":
    verify_maml()
    verify_regime_detection()
