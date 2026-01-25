"""
Learning Vector Quantization (LVQ) Module.

This module implements the LVQ algorithm, a prototype-based classification model.
"""

import torch
from torch import nn


class LVQ(nn.Module):
    """
    Learning Vector Quantization (LVQ) implementation.
    A prototype-based supervised classification algorithm.
    """

    prototype_labels: torch.Tensor

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        prototypes_per_class: int = 1,
        output_type: str = "prediction",
    ) -> None:
        """
        Initialize the LVQ model.

        Args:
            input_dim (int): Dimensionality of the input features.
            num_classes (int): Number of target classes.
            prototypes_per_class (int, optional): Number of prototypes representing each class. Defaults to 1.
            output_type (str, optional): Format of the output.
                'prediction' returns class indices, 'logits' returns negative distances. Defaults to "prediction".
        """
        super().__init__()
        self.output_type = output_type
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.prototypes_per_class = prototypes_per_class
        self.num_prototypes = num_classes * prototypes_per_class

        # Initialize prototypes roughly around 0
        self.prototypes = nn.Parameter(torch.randn(self.num_prototypes, input_dim))

        # Assign labels to prototypes (e.g., first 10 for class 0, next 10 for class 1)
        # We model this as a fixed property, not a parameter
        self.register_buffer(
            "prototype_labels",
            torch.arange(num_classes).repeat_interleave(prototypes_per_class),
        )

    def forward(self, x: torch.Tensor, return_embedding: bool = False) -> torch.Tensor:
        """
        Forward pass finds the nearest prototype.
        In inference, this performs classification.
        In training, custom training loop is needed for LVQ updates,
        or we can approximate with soft competitive learning.
        """
        device = x.device

        # Handle sequence: take last step
        if x.ndim == 3:
            x = x[:, -1, :]

        # Calculate distances: (Batch, Prototypes)
        # ||x - p||^2 = ||x||^2 + ||p||^2 - 2 <x, p>
        dists = torch.cdist(x, self.prototypes, p=2)

        # Find nearest prototype
        _min_dists, _indices = torch.min(dists, dim=1)

        if return_embedding:
            # Embedding is the distance vector to prototypes
            return dists

        # Prediction is the label of the nearest prototype
        # We can simulate soft-output for compatibility
        # Convert distances to "probabilities" via softmax of negative distances
        logits = -dists

        # But we need to aggregate by class
        # (Batch, Prototypes) -> (Batch, Classes)
        # We max over prototypes of the same class
        class_logits = []
        for c in range(self.num_classes):
            mask = torch.eq(self.prototype_labels, c)
            if torch.any(mask):
                class_logits.append(torch.max(logits[:, mask], dim=1)[0])
            else:
                class_logits.append(torch.full((x.shape[0],), -1e9, device=device))

        class_logits_tensor = torch.stack(class_logits, dim=1)

        # For prediction, we return indices
        if self.output_type == "prediction":
            return torch.argmax(class_logits_tensor, dim=1, keepdim=True).float()

        return class_logits_tensor

    def training_step(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        LVQ specific training step (LVQ1 rule).
        This updates prototypes manually as LVQ is non-gradient based primarily.
        However, for PyTorch compatibility, we might expose a loss.
        """
        # Standard LVQ is better for custom optimizers,
        # but here's a GLVQ (Generalized LVQ) differentiable approximation.
        # Loss = (d+ - d-) / (d+ + d-)
        # where d+ is dist to nearest proto of CORRECT class
        # and d- is dist to nearest proto of INCORRECT class

        device = x.device
        if x.ndim == 3:
            x = x[:, -1, :]
        if y.ndim == 2:
            y = y.squeeze(-1)

        y = y.long()

        dists = torch.cdist(x, self.prototypes, p=2)

        losses = []
        for i in range(x.shape[0]):
            target = y[i]

            # Mask for correct class prototypes
            correct_mask = self.prototype_labels == target
            if not correct_mask.any():
                continue

            # d+
            d_plus = torch.min(dists[i, correct_mask])

            # Mask for incorrect class prototypes
            incorrect_mask = ~correct_mask
            if not incorrect_mask.any():
                continue

            # d-
            d_minus = torch.min(dists[i, incorrect_mask])

            # GLVQ Loss
            # Using sigmoid for smoother gradients or the standard ratio
            loss = (d_plus - d_minus) / (d_plus + d_minus + 1e-8)
            losses.append(loss)

        if not losses:
            return torch.tensor(0.0, device=device, requires_grad=True)

        return torch.stack(losses).mean()
