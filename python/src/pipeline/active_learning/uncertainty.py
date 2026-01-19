"""
Uncertainty Estimation for Active Learning.

This module provides tools to estimate predictive uncertainty, which can be 
used to sample the most informative data points for training.
"""

import torch
import torch.nn as nn
from typing import Dict, List

def pinball_loss(pred: torch.Tensor, target: torch.Tensor, quantiles: List[float]) -> torch.Tensor:
    """
    Computes the pinball loss for quantile regression.
    
    Args:
        pred: Predicted values [Batch, len(quantiles)]
        target: Ground truth values [Batch, 1]
        quantiles: List of quantiles (e.g., [0.1, 0.5, 0.9])
    """
    losses = []
    for i, q in enumerate(quantiles):
        errors = target - pred[:, i:i+1]
        loss = torch.max((q - 1) * errors, q * errors)
        losses.append(loss.mean())
    return torch.stack(losses).mean()


class QuantileHead(nn.Module):
    """
    Prediction head for Quantile Regression.
    Outputs multiple quantiles for uncertainty estimation.
    """
    def __init__(self, input_dim: int, quantiles: List[float]):
        super().__init__()
        self.quantiles = quantiles
        self.head = nn.Linear(input_dim, len(quantiles))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


def mc_dropout_predict(
    model: nn.Module, 
    x: torch.Tensor, 
    n_samples: int = 10
) -> Dict[str, torch.Tensor]:
    """
    Perform Monte Carlo Dropout prediction.
    
    Args:
        model: Model with dropout layers.
        x: Input batch.
        n_samples: Number of forward passes with dropout enabled.
        
    Returns:
        Dict containing mean and variance (uncertainty score).
    """
    model.train() # Enable dropout
    preds = []
    
    with torch.no_grad():
        for _ in range(n_samples):
            preds.append(model(x))
            
    preds_stack = torch.stack(preds) # [n_samples, batch, output_dim]
    mean = preds_stack.mean(dim=0)
    variance = preds_stack.var(dim=0)
    
    return {
        "mean": mean,
        "variance": variance,
        "std": torch.sqrt(variance)
    }
