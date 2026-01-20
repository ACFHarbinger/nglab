"""
Association rule learning models for NGLab.
"""

import torch

from ..mac.base import ClassicalModel
from .association_rule_learning.apriori import AprioriAlgorithm
from .association_rule_learning.eclat import EclatAlgorithm
from .association_rule_learning.fpgrowth import FPGrowthAlgorithm


class AssociationRuleModel(ClassicalModel):
    """Base class for association rule models."""

    def __init__(self, **kwargs):
        super().__init__(output_type="rules")

    def forward(self, x, **kwargs):
        """
        For association rules, forward might return the rules or matched rules for x.
        For now, returns a dummy tensor to comply with interface.
        """
        device = x.device
        return torch.zeros((x.shape[0], 1)).to(device)

    def get_rules(self):
        if self.model and self._is_fitted:
            return self.model.rules
        return []


class AprioriModel(AssociationRuleModel):
    def __init__(self, min_support=0.5, min_confidence=0.7, **kwargs):
        super().__init__()
        self.model = AprioriAlgorithm(
            min_support=min_support, min_confidence=min_confidence, **kwargs
        )


class FPGrowthModel(AssociationRuleModel):
    def __init__(self, min_support=0.5, min_confidence=0.7, **kwargs):
        super().__init__()
        self.model = FPGrowthAlgorithm(
            min_support=min_support, min_confidence=min_confidence, **kwargs
        )


class EclatModel(AssociationRuleModel):
    def __init__(self, min_support=0.5, min_confidence=0.7, **kwargs):
        super().__init__()
        self.model = EclatAlgorithm(
            min_support=min_support, min_confidence=min_confidence, **kwargs
        )
