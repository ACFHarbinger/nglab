"""
Active Learning Selection Samplers.

This module provides strategies for selecting the most informative samples
from an unlabelled pool based on uncertainty scores.
"""

from typing import Any

import numpy as np
import torch


class BaseSampler:
    """Base class for Active Learning samplers."""

    def __init__(self, budget: int):
        """Initialize the sampler with a selection budget."""
        self.budget = budget

    def select(self, scores: torch.Tensor) -> np.ndarray[Any, Any]:
        """
        Select indices based on scores.

        Args:
            scores: Uncertainty scores [PoolSize]

        Returns:
            np.ndarray: Indices of selected samples.
        """
        raise NotImplementedError


class UncertaintySampler(BaseSampler):
    """Selects samples with the highest uncertainty scores."""

    def select(self, scores: torch.Tensor) -> np.ndarray[Any, Any]:
        """Select samples with highest uncertainty scores."""
        # scores: higher is more uncertain
        _, indices = torch.topk(scores.view(-1), k=self.budget)
        return indices.cpu().numpy()


class EntropySampler(BaseSampler):
    """
    Selects samples with the highest predictive entropy.
    Suitable for classification tasks.
    """

    def select(self, probs: torch.Tensor) -> np.ndarray[Any, Any]:
        """
        Args:
            probs: Predicted probabilities [PoolSize, NumClasses]
        """
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
        _, indices = torch.topk(entropy, k=self.budget)
        return indices.cpu().numpy()


class BaldSampler(BaseSampler):
    """
    Bayesian Active Learning by Disagreement (BALD).
    Selects samples that maximize the information gain about model parameters.
    """

    def select(self, mc_preds: torch.Tensor) -> np.ndarray[Any, Any]:
        """
        Args:
            mc_preds: MC Dropout predictions [num_samples, pool_size, num_classes]
        """
        # Average of entropy
        entropy_of_avg = -torch.sum(
            mc_preds.mean(0) * torch.log(mc_preds.mean(0) + 1e-10), dim=1
        )

        # Entropy of average
        avg_of_entropy = -torch.mean(
            torch.sum(mc_preds * torch.log(mc_preds + 1e-10), dim=2), dim=0
        )

        bald_score = entropy_of_avg - avg_of_entropy
        _, indices = torch.topk(bald_score, k=self.budget)
        return indices.cpu().numpy()


class RandomSampler(BaseSampler):
    """Baseline: Selects samples randomly."""

    def select(self, scores: torch.Tensor) -> np.ndarray[Any, Any]:
        """Select samples randomly."""
        pool_size = scores.size(0)
        indices = np.random.choice(pool_size, size=self.budget, replace=False)
        return indices
