"""
Diffusion Lightning Module for Time Series Forecasting (DDPM).
"""

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

    def __init__(self, model: nn.Module, cfg: dict):
        """
        Args:
            model (nn.Module): The noise prediction model (e.g. DiffusionUNet1D).
                               Expects forward(x_t, t, cond).
            cfg (dict): config
        """
        super().__init__(cfg)
        self.model = model

        # DDPM Constants
        self.timesteps = cfg.get("timesteps", 1000)
        self.beta_start = cfg.get("beta_start", 0.0001)
        self.beta_end = cfg.get("beta_end", 0.02)

        # Define schedule
        # Linear schedule
        self.register_buffer(
            "betas", torch.linspace(self.beta_start, self.beta_end, self.timesteps)
        )
        self.register_buffer("alphas", 1.0 - self.betas)
        self.register_buffer("alphas_cumprod", torch.cumprod(self.alphas, dim=0))
        self.register_buffer(
            "alphas_cumprod_prev", F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        )
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / self.alphas))

        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(self.alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - self.alphas_cumprod)
        )

        self.register_buffer(
            "posterior_variance",
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod),
        )

    def q_sample(self, x_start, t, noise=None):
        """
        Forward diffusion process: q(x_t | x_0).
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][
            :, None, None
        ]  # Broadcast to (B, L, F)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][
            :, None, None
        ]

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def training_step(self, batch, batch_idx):
        """
        Training: Minimize MSE between predicted noise and added noise.
        """
        if isinstance(batch, dict):
            # Cond = Observation (History), Target = Future
            cond = batch.get("observation")
            target = batch.get("target")
        else:
            cond, target = batch

        batch_size = target.size(0)

        # Sample time step t
        t = torch.randint(0, self.timesteps, (batch_size,), device=self.device).long()

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

    def validation_step(self, batch, batch_idx):
        """
        Validation: Compute loss.
        """
        loss = self.training_step(batch, batch_idx)
        self.log("val/diffusion_loss", loss, prog_bar=True)
        return loss

    @torch.no_grad()
    def p_sample(self, x, t, t_index, cond):
        """
        Reverse step: p(x_{t-1} | x_t).
        """
        betas_t = self.betas[t]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t]

        # Reshape for broadcasting
        # Assuming x is (B, L, F)
        batch_size = x.size(0)

        # Predict noise
        # Need to ensure t is a tensor of shape (B,) with value t_index
        t_tensor = torch.full(
            (batch_size,), t_index, device=self.device, dtype=torch.long
        )

        model_mean = self.model(x, t_tensor, cond)

        # Equation: x_{t-1} = 1/sqrt(alpha) * (x_t - beta/sqrt(1-alpha_bar) * eps_theta)

        # Reshape coeffs to (1, 1, 1) or (B, 1, 1) ??
        # The buffers are full size arrays [T]. We indexed scalar t.
        # But wait, self.betas[t] is a scalar (tensor(0.02)).
        # We need to broadcast.

        coeff = betas_t / sqrt_one_minus_alphas_cumprod_t

        pred_mean = sqrt_recip_alphas_t * (x - coeff * model_mean)

        if t_index == 0:
            return pred_mean
        else:
            posterior_variance_t = self.posterior_variance[t]
            noise = torch.randn_like(x)
            return pred_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample(self, cond):
        """
        Generate samples given condition (history).
        """
        # Shape: Matches cond shape or target shape?
        # Typically forecasting target has same features as cond (or subset) and defined horizon.
        # If output_dim is defined in model, use that.

        batch_size = cond.size(0)
        # We need the target length. Usually defined in cfg or passed
        pred_len = self.cfg.get("pred_len", 1)
        # Or derive from model output_dim / feature_dim?
        # Model output_dim is channels. Length is Sequence Length?
        # UNet usually preserves length L.
        # If we predict L future steps, we need noise of shape (B, L, F).
        # We assume pred_len is the Sequence Length of the UNet's processing window.

        # NOTE: If we use UNet1D, it outputs same length as input.
        # So we generate 'pred_len' worth of noise.

        # Get shape from model output config or cond?
        # Let's assume we predict 'pred_len' steps
        output_dim = self.model.output_dim

        device = self.device
        img = torch.randn((batch_size, pred_len, output_dim), device=device)

        for i in reversed(range(0, self.timesteps)):
            img = self.p_sample(img, i, i, cond)

        return img
