"""
Reinforcement Learning Module for NGLab.

Integrates TorchRL components with PyTorch Lightning to provide a scalable
training loop for RL agents (PPO and variants).
"""

from collections.abc import Callable
from typing import Any, cast

import torch
from torch import nn
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.objectives import ClipPPOLoss, ValueEstimators

from .base import BaseModule


class RLLightningModule(BaseModule):
    """
    Lightning Module for Reinforcement Learning (PPO).
    Manages the interaction between Policy, Environment (Collection), and Loss updates.
    """

    def __init__(
        self,
        agent_module: nn.Module,
        value_module: nn.Module,
        env_maker: Callable[[], Any],
        cfg: dict[str, Any],
    ) -> None:
        """
        Initialize the RL module.

        Args:
            agent_module (nn.Module): The policy network (actor).
            value_module (nn.Module): The value network (critic).
            env_maker (callable): A function that returns a wrapped environment.
            cfg (Dict): Configuration parameters.
        """
        super().__init__(cfg)
        self.save_hyperparameters(ignore=["agent_module", "value_module", "env_maker"])

        self.agent = agent_module
        self.critic = value_module
        self.env_maker = env_maker

        # Create collector
        # Note: In a real PL loop, collection happens often in 'training_step' or via a DataModule.
        # Here we embed the collector loop to align with TorchRL idioms or typical PPO steps.

        # Loss Module
        self.loss_module = ClipPPOLoss(
            actor_network=cast(Any, self.agent),
            critic_network=cast(Any, self.critic),
            clip_epsilon=cfg.get("clip_epsilon", 0.2),
            entropy_bonus=bool(cfg.get("ent_coef", 0.0)),
            gamma=cfg.get("gamma", 0.99),
            gae_lambda=cfg.get("gae_lambda", 0.95),
            loss_critic_type="l2_smooth",
        )
        self.loss_module.set_keys(advantage="advantage", value_target="value_target")
        self.loss_module.make_value_estimator(ValueEstimators.GAE)

        self.frames_per_batch = int(cfg.get("frames_per_batch", 1000))
        self.total_frames = int(cfg.get("total_frames", 1_000_000))
        self.ppo_epochs = int(cfg.get("ppo_epochs", 10))
        self.automatic_optimization = False

        # Replay Buffer
        self.replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(max_size=self.frames_per_batch),
            batch_size=int(cfg.get("mini_batch_size", 64)),
        )

    def setup(self, stage: str | None = None) -> None:
        """
        Initialize the data collector.
        """
        # Create the collector here or in simple training loop
        # For PL, we usually iterate over a DataLoader.
        # But PPO is on-policy.
        # Option: Make the DataCollector an IterableDataset.
        device = cast(torch.device, self.device)
        self.collector = SyncDataCollector(
            self.env_maker(),
            self.agent,
            frames_per_batch=self.frames_per_batch,
            total_frames=self.total_frames,
            split_trajs=False,
            device=device,
        )

    def train_dataloader(self) -> Any:
        """
        Return the data collector as the training data source.
        """
        # Return the collector as the dataloader source
        return self.collector

    def training_step(self, batch: Any, batch_idx: int) -> None:  # type: ignore[override]
        """
        Perform a PPO training step on a collected batch.
        """
        # 'batch' here is a TensorDict with 'frames_per_batch' steps collected

        # 1. PPO requires updates on this batch for multiple epochs
        # PPO is slightly tricky in PL standard loop because of inner epochs on the same batch.
        # We can simulate this by manual backward or just doing one gradient step if configured differently.
        # Standard PPO: Iterate K epochs on this collected batch.

        # Calculate Advantages
        with torch.no_grad():
            self.loss_module.value_estimator(
                batch,
                params=self.loss_module.critic_network_params,
                target_params=self.loss_module.target_critic_network_params,
            )

        # Flatten batch for mini-batch update
        batch = batch.reshape(-1)
        self.replay_buffer.extend(batch)
        
        # Access optimizer correctly
        optimizers = self.optimizers()
        if isinstance(optimizers, list):
            opt = optimizers[0]
        else:
            opt = optimizers

        # Inner PPO Loop
        total_loss = torch.tensor(0.0, device=cast(torch.device, self.device))
        for _ in range(self.ppo_epochs):
            for _i, sub_batch in enumerate(self.replay_buffer):
                loss_vals = self.loss_module(sub_batch)
                loss_value = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )

                # Manual Optimization (if Automatic is disabled) or we accumulate
                # Since PL expects 1 loss per step, this nested loop is unusual.
                # Simplification: We return the average loss of the last epoch for logging,
                # but we must perform optimization steps here manually OR
                # use Automatic Optimization and just do one pass?
                # Best practice in PL for PPO is manual optimization.

                opt_any: Any = opt
                opt_any.zero_grad()
                self.manual_backward(loss_value)
                opt_any.step()

                total_loss += loss_value.detach()

        # Clear buffer after update
        # ReplayBuffer is circular/lazy, but for PPO we flush it effectively by overwriting next time
        # or we just used it for easy minibatch sampling.

        avg_loss = total_loss / (self.ppo_epochs * len(self.replay_buffer))
        self.log("train/loss", avg_loss)
