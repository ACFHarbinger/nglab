import logging

import torch
from torch.utils.data import DataLoader, Dataset

from python.src.pipeline.hpo.optimize import bayesian_optimization, run_dehb_search

logging.basicConfig(level=logging.INFO)


class DummyDataset(Dataset):
    def __init__(self, size=100, seq_len=30, dim=10):
        # Observation is expected as {"observation": ..., "target": ...} by SLLightningModule
        # LSTM expects [batch, seq_len, input_dim]
        self.data = torch.randn(size, seq_len, dim)
        self.labels = torch.randn(size, 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {"observation": self.data[idx], "target": self.labels[idx]}


def train_loader_factory():
    return DataLoader(DummyDataset(), batch_size=16, shuffle=True)


def val_loader_factory():
    return DataLoader(DummyDataset(), batch_size=16, shuffle=False)


def verify_optimize():
    opts = {
        "model_cfg": {
            "name": "LSTM",
            "feature_dim": 10,
            "output_dim": 1,
            "n_layers": 1,
        },
        "train_loader_factory": train_loader_factory,
        "val_loader_factory": val_loader_factory,
        "max_epochs": 1,
        "run_name": "verify_hpo_optuna",
        "save_plots": False,
        "problem": "wcvrp",  # For get_config_space in DEHB
        "hop_range": [1e-5, 1e-2],
        "output_dir": "results_verify",
    }

    print("Starting Bayesian Optimization (Optuna)...")
    best_params_optuna = bayesian_optimization(opts, n_trials=2)
    print(f"Best parameters (Optuna): {best_params_optuna}")

    print("\nStarting DEHB Optimization...")
    best_params_dehb = run_dehb_search(opts, fevals=2, max_fidelity=2)
    print(f"Best parameters (DEHB): {best_params_dehb}")


if __name__ == "__main__":
    verify_optimize()
