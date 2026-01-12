import torch
import pytorch_lightning as pl
from torchrl.collectors import SyncDataCollector
from torchrl.data import ReplayBuffer, LazyTensorStorage
from torchrl.objectives import ClipPPOLoss, ValueEstimators
from tensordict import TensorDict

class RLLightningModule(pl.LightningModule):
    """
    Lightning Module for Reinforcement Learning (PPO).
    Manages the interaction between Policy, Environment (Collection), and Loss updates.
    """
    def __init__(self, agent_module, value_module, env_maker, cfg):
        super().__init__()
        self.save_hyperparameters(ignore=['agent_module', 'value_module', 'env_maker'])
        self.cfg = cfg
        
        self.agent = agent_module
        self.critic = value_module
        self.env_maker = env_maker
        
        # Create collector
        # Note: In a real PL loop, collection happens often in 'training_step' or via a DataModule.
        # Here we embed the collector loop to align with TorchRL idioms or typical PPO steps.
        
        # Loss Module
        self.loss_module = ClipPPOLoss(
            actor_network=self.agent,
            critic_network=self.critic,
            clip_epsilon=cfg.get('clip_epsilon', 0.2),
            entropy_bonus=bool(cfg.get('ent_coef', 0.0)),
            gamma=cfg.get('gamma', 0.99),
            gae_lambda=cfg.get('gae_lambda', 0.95),
            loss_critic_type="l2_smooth"
        )
        self.loss_module.set_keys(advantage="advantage", value_target="value_target")
        self.loss_module.make_value_estimator(ValueEstimators.GAE)
        
        self.frames_per_batch = cfg.get('frames_per_batch', 1000)
        self.total_frames = cfg.get('total_frames', 1_000_000)
        self.ppo_epochs = cfg.get('ppo_epochs', 10)
        
        # Replay Buffer
        self.replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(max_size=self.frames_per_batch),
            batch_size=cfg.get('mini_batch_size', 64)
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.loss_module.parameters(), lr=self.cfg.get('learning_rate', 3e-4))
        return optimizer

    def setup(self, stage=None):
        # Create the collector here or in simple training loop
        # For PL, we usually iterate over a DataLoader. 
        # But PPO is on-policy.
        # Option: Make the DataCollector an IterableDataset.
        self.collector = SyncDataCollector(
            self.env_maker(),
            self.agent,
            frames_per_batch=self.frames_per_batch,
            total_frames=self.total_frames,
            split_trajs=False,
            device=self.device
        )

    def train_dataloader(self):
        # Return the collector as the dataloader source
        return self.collector

    def training_step(self, batch, batch_idx):
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
        
        # Inner PPO Loop
        total_loss = 0
        for _ in range(self.ppo_epochs):
            for i, sub_batch in enumerate(self.replay_buffer):
                loss_vals = self.loss_module(sub_batch)
                loss_value = loss_vals["loss_objective"] + \
                             loss_vals["loss_critic"] + \
                             loss_vals["loss_entropy"]
                
                # Manual Optimization (if Automatic is disabled) or we accumulate
                # Since PL expects 1 loss per step, this nested loop is unusual.
                # Simplification: We return the average loss of the last epoch for logging,
                # but we must perform optimization steps here manually OR 
                # use Automatic Optimization and just do one pass?
                # Best practice in PL for PPO is manual optimization.
                
                opt = self.optimizers()
                opt.zero_grad()
                self.manual_backward(loss_value)
                opt.step()
                
                total_loss += loss_value.detach()

        # Clear buffer after update
        # ReplayBuffer is circular/lazy, but for PPO we flush it effectively by overwriting next time
        # or we just used it for easy minibatch sampling.
        
        self.log('train/loss', total_loss / (self.ppo_epochs * len(self.replay_buffer)))
        return None # We handled optimization manually

    @property
    def automatic_optimization(self) -> bool:
        return False
