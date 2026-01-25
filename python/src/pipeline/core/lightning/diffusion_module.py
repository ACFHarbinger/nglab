"""
Diffusion Lightning Module for Time Series Forecasting (DDPM).
"""

from typing import Any, cast

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from .base import BaseModule


class DiffusionLightningModule(BaseModule):
    """
    Lightning Module for Denoising Diffusion Probabilistic Models (DDPM).
    Training: Predict noise added to target y_0 given condition x and time t.
    Sampling: Reverse diffusion from noise to y_0 conditioned on x.
    """

    def __init__(self, model: nn.Module, cfg: dict[str, Any]) -> None:
        """
        Args:
            model (nn.Module): The noise prediction model (e.g. DiffusionUNet1D).
                               Expects forward(x_t, t, cond).
            cfg (dict): config
        """
        super().__init__(cfg)
        self.model = model

        # DDPM Constants
        self.timesteps = int(cfg.get("timesteps", 1000))
        self.beta_start = float(cfg.get("beta_start", 0.0001))
        self.beta_end = float(cfg.get("beta_end", 0.02))

        # Define schedule
        # Linear schedule
        self.register_buffer(
            "betas", torch.linspace(self.beta_start, self.beta_end, self.timesteps)
        )
        self.betas: torch.Tensor

        self.register_buffer("alphas", 1.0 - self.betas)
        self.alphas: torch.Tensor

        self.register_buffer("alphas_cumprod", torch.cumprod(self.alphas, dim=0))
        self.alphas_cumprod: torch.Tensor

        self.register_buffer(
            "alphas_cumprod_prev", F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        )
        self.alphas_cumprod_prev: torch.Tensor

        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / self.alphas))
        self.sqrt_recip_alphas: torch.Tensor

        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(self.alphas_cumprod))
        self.sqrt_alphas_cumprod: torch.Tensor

        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - self.alphas_cumprod)
        )
        self.sqrt_one_minus_alphas_cumprod: torch.Tensor

        self.register_buffer(
            "posterior_variance",
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod),
        )
        self.posterior_variance: torch.Tensor

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward diffusion process: q(x_t | x_0).
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        # Broadcast parameters to match (B, L, F)
        # t is (B,) long tensor
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][
            :, None, None
        ]  # Broadcast to (B, L, F)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][
            :, None, None
        ]

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """
        Training: Minimize MSE between predicted noise and added noise.
        """
        if isinstance(batch, dict):
            # Cond = Observation (History), Target = Future
            cond = batch.get("observation")
            target = batch.get("target")
        else:
            cond, target = batch

        if target is None:
            raise ValueError("Target is None in training batch")

        batch_size = target.size(0)
        device = cast(torch.device, self.device)

        # Sample time step t
        t = torch.randint(0, self.timesteps, (batch_size,), device=device).long()

        # Noise
        noise = torch.randn_like(target)

        # Noisy target
        x_t = self.q_sample(target, t, noise)

        # Predict noise
        # Model signature: forward(x_noisy, t, condition)
        predicted_noise = self.model(x_t, t, cond=cond)

        loss = F.mse_loss(predicted_noise, noise)

        self.log("train/diffusion_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """
        Validation: Compute loss.
        """
        loss = self.training_step(batch, batch_idx)
        self.log("val/diffusion_loss", loss, prog_bar=True)
        return loss

    @torch.no_grad()
    def p_sample(
        self, x: torch.Tensor, t: int, t_index: int, cond: torch.Tensor
    ) -> torch.Tensor:
        """
        Reverse step: p(x_{t-1} | x_t).
        """
        betas_t: torch.Tensor = self.betas[t]
        sqrt_one_minus_alphas_cumprod_t: torch.Tensor = (
            self.sqrt_one_minus_alphas_cumprod[t]
        )
        sqrt_recip_alphas_t: torch.Tensor = self.sqrt_recip_alphas[t]

        # Reshape for broadcasting
        # Assuming x is (B, L, F)
        batch_size = x.size(0)
        device = cast(torch.device, self.device)

        # Predict noise
        # Need to ensure t is a tensor of shape (B,) with value t_index
        t_tensor = torch.full((batch_size,), t_index, device=device, dtype=torch.long)

        model_mean = cast(torch.Tensor, self.model(x, t_tensor, cond))

        # Equation: x_{t-1} = 1/sqrt(alpha) * (x_t - beta/sqrt(1-alpha_bar) * eps_theta)

        coeff = betas_t / sqrt_one_minus_alphas_cumprod_t

        pred_mean = sqrt_recip_alphas_t * (x - coeff * model_mean)

        if t_index == 0:
            return pred_mean
        else:
            posterior_variance_t = self.posterior_variance[t]
            noise = torch.randn_like(x)
            return pred_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample(self, cond: torch.Tensor) -> torch.Tensor:
        """
        Generate samples given condition (history).
        """
        batch_size = cond.size(0)
        # We need the target length. Usually defined in cfg or passed
        pred_len = int(self.cfg.get("pred_len", 1))

        # Get device
        device = cast(torch.device, self.device)

        # Determine output dim
        # Try to infer from model if possible, usually last linear layer out_features / forecast_horizon
        # But here we might just assume input_dim == output_dim if UNet1D
        if hasattr(self.model, "output_dim"):
            output_dim = int(getattr(self.model, "output_dim", 1))
        else:
            output_dim = 1  # Fallback

        img = torch.randn((batch_size, pred_len, output_dim), device=device)

        for i in reversed(range(0, self.timesteps)):
            img = self.p_sample(img, i, i, cond)

        return img
