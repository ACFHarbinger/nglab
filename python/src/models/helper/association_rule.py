
"""
Association rule learning models for NGLab.
"""

from typing import Any, cast

import torch

from ..mac.base import ClassicalModel
from .association_rule_learning.apriori import AprioriAlgorithm
from .association_rule_learning.eclat import EclatAlgorithm
from .association_rule_learning.fpgrowth import FPGrowthAlgorithm


class AssociationRuleModel(ClassicalModel):
    """Base class for association rule models."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize Association Rule Model."""
        super().__init__(output_type="rules")
        self.model: Any | None = None

    def forward(
        self,
        x: torch.Tensor,
        return_embedding: bool | None = None,
        return_sequence: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass. Returns a dummy tensor to comply with interface.
        """
        device = x.device
        return torch.zeros((x.shape[0], 1)).to(device)

    def get_rules(self) -> list[dict[str, Any]]:
        """Return the learned rules."""
        if self.model and self._is_fitted:
            return cast(list[dict[str, Any]], self.model.rules)
        return []


class AprioriModel(AssociationRuleModel):
    """Apriori Association Rule Model."""

    def __init__(
        self, min_support: float = 0.5, min_confidence: float = 0.7, **kwargs: Any
    ) -> None:
        """Initialize Apriori Model."""
        super().__init__()
        self.model = AprioriAlgorithm(
            min_support=min_support, min_confidence=min_confidence
        )


class FPGrowthModel(AssociationRuleModel):
    """FP-Growth Association Rule Model."""

    def __init__(
        self, min_support: float = 0.5, min_confidence: float = 0.7, **kwargs: Any
    ) -> None:
        """Initialize FP-Growth Model."""
        super().__init__()
        self.model = FPGrowthAlgorithm(
            min_support=min_support, min_confidence=min_confidence
        )


class EclatModel(AssociationRuleModel):
    """Eclat Association Rule Model."""

    def __init__(
        self, min_support: float = 0.5, min_confidence: float = 0.7, **kwargs: Any
    ) -> None:
        """Initialize Eclat Model."""
        super().__init__()
        self.model = EclatAlgorithm(
            min_support=min_support, min_confidence=min_confidence, **kwargs
        )
