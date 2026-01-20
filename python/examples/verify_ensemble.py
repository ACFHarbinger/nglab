"""
Verification script for Ensemble Models.
"""

import torch

from python.src.models.ensemble import create_ensemble_from_configs


def verify_ensemble():
    print("--- Ensemble Model Verification ---")

    # Create ensemble from multiple model configs
    # NOTE: All models must produce the same output_dim for stacking
    configs = [
        {
            "name": "LSTM",
            "feature_dim": 10,
            "hidden_dim": 32,
            "num_layers": 1,
            "output_dim": 8,
            "seq_len": 20,
            "output_type": "prediction",
        },
        {
            "name": "LSTM",
            "feature_dim": 10,
            "hidden_dim": 64,
            "num_layers": 2,
            "output_dim": 8,
            "seq_len": 20,
            "output_type": "prediction",
        },
        {
            "name": "GRU",
            "feature_dim": 10,
            "hidden_dim": 48,
            "num_layers": 1,
            "output_dim": 8,
            "seq_len": 20,
            "output_type": "prediction",
        },
    ]

    print(f"Creating ensemble with {len(configs)} models...")
    ensemble = create_ensemble_from_configs(configs, strategy="average")
    print(f"Ensemble created: {ensemble.n_models} models")

    # Test forward pass
    batch_size = 8
    seq_len = 20
    feature_dim = 10
    x = torch.randn(batch_size, seq_len, feature_dim)

    print("\nTesting forward pass...")
    output = ensemble(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    # Test uncertainty estimation
    print("\nTesting uncertainty estimation...")
    uncertainty = ensemble.predict_with_uncertainty(x)
    print(f"Mean shape: {uncertainty['mean'].shape}")
    print(f"Std shape: {uncertainty['std'].shape}")
    print(f"Predictions shape: {uncertainty['predictions'].shape}")

    # Test weighted averaging
    print("\nTesting weighted ensemble...")
    weights = [0.5, 0.3, 0.2]
    weighted_ensemble = create_ensemble_from_configs(
        configs, strategy="weighted", weights=weights
    )
    weighted_output = weighted_ensemble(x)
    print(f"Weighted output shape: {weighted_output.shape}")

    print("\n✓ All ensemble tests passed!")


if __name__ == "__main__":
    verify_ensemble()
