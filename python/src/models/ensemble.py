"""
Ensemble Model Wrapper for Robust Predictions.

Combines multiple models using various strategies:
- Simple averaging
- Weighted averaging
- Voting (for classification)
- Stacking (meta-learner)
"""

from collections.abc import Sequence
from typing import Any, Literal, cast

import torch
from torch import nn

from .time_series import TimeSeriesBackbone


class EnsembleModel(nn.Module):
    """
    Ensemble wrapper that combines predictions from multiple models.

    Supports various aggregation strategies for improved robustness.
    """

    def __init__(
        self,
        models: Sequence[nn.Module],
        strategy: Literal["average", "weighted", "voting", "stacking"] = "average",
        weights: list[float] | None = None,
        meta_learner: nn.Module | None = None,
    ) -> None:
        """
        Initialize ensemble.

        Args:
            models: List of base models to combine.
            strategy: Aggregation strategy.
            weights: Optional weights for weighted averaging.
            meta_learner: Optional meta-learner for stacking.
        """
        super().__init__()
        self.models = nn.ModuleList(models)
        self.strategy = strategy
        self.n_models = len(models)

        if weights is not None:
            self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))
        else:
            self.register_buffer("weights", torch.ones(self.n_models) / self.n_models)
        self.weights: torch.Tensor

        self.meta_learner = meta_learner

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ensemble.

        Args:
            x: Input tensor.

        Returns:
            Aggregated predictions.
        """
        predictions = []
        for model in self.models:
            cm = torch.no_grad() if not self.training else torch.enable_grad()
            with cm:
                pred = model(x)
                predictions.append(pred)

        # Stack predictions: [n_models, batch, output_dim]
        stacked = torch.stack(predictions, dim=0)

        if self.strategy == "average":
            return stacked.mean(dim=0)

        elif self.strategy == "weighted":
            # Weighted average
            # Use explicit cast or type hint for self.weights
            weights = self.weights.view(-1, 1, 1)
            return (stacked * weights).sum(dim=0)

        elif self.strategy == "voting":
            # Hard voting (for classification)
            votes = stacked.argmax(dim=-1)  # [n_models, batch]
            # Mode across models
            return votes.mode(dim=0).values

        elif self.strategy == "stacking":
            if self.meta_learner is None:
                raise ValueError("Stacking requires a meta_learner")
            # Concatenate predictions as features for meta-learner
            concat = stacked.permute(1, 0, 2).flatten(
                start_dim=1
            )  # [batch, n_models * output_dim]
            return cast(torch.Tensor, self.meta_learner(concat))

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def predict_with_uncertainty(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Get predictions along with uncertainty estimates from ensemble disagreement.

        Args:
            x: Input tensor.

        Returns:
            Dict with 'mean', 'std', and 'predictions'.
        """
        self.eval()
        predictions = []
        for model in self.models:
            with torch.no_grad():
                pred = model(x)
                predictions.append(pred)

        stacked = torch.stack(predictions, dim=0)

        return {
            "mean": stacked.mean(dim=0),
            "std": stacked.std(dim=0),
            "predictions": stacked,
        }


def create_ensemble_from_configs(
    configs: list[dict[str, Any]],
    strategy: str = "average",
    weights: list[float] | None = None,
) -> EnsembleModel:
    """
    Factory function to create an ensemble from a list of model configs.

    Args:
        configs: List of model configuration dictionaries.
        strategy: Aggregation strategy.
        weights: Optional weights.

    Returns:
        EnsembleModel instance.
    """
    # Cast to Sequence[nn.Module] to satisfy invariance
    models: Sequence[nn.Module] = [TimeSeriesBackbone(cfg) for cfg in configs]

    # Cast strategy string to Literal
    strat_literal = cast(
        Literal["average", "weighted", "voting", "stacking"],
        (
            strategy
            if strategy in ["average", "weighted", "voting", "stacking"]
            else "average"
        ),
    )

    return EnsembleModel(models, strategy=strat_literal, weights=weights)
