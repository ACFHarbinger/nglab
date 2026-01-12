"""
Meta-Learning Wrapper for NGLab.

Implements Model-Agnostic Meta-Learning (MAML) using the 'higher' library,
enabling fast adaptation of policies to new market conditions or tasks.
"""
import torch
import higher
from pytorch_lightning import LightningModule

class MAMLWrapper(LightningModule):
    """
    Model-Agnostic Meta-Learning (MAML) Wrapper.
    Wraps an inner LightningModule (e.g. RLModule) to perform meta-learning updates.
    """
    def __init__(self, inner_module: LightningModule, inner_lr=0.01, meta_lr=0.001, num_inner_steps=1):
        """
        Initialize the MAML wrapper.

        Args:
            inner_module (LightningModule): The module to be adapted.
            inner_lr (float): Learning rate for the inner adaptation loop.
            meta_lr (float): Learning rate for the outer meta-update loop.
            num_inner_steps (int): Number of adaptation steps.
        """
        super().__init__()
        self.save_hyperparameters(ignore=['inner_module'])
        self.module = inner_module
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.num_inner_steps = num_inner_steps

    def configure_optimizers(self):
        """
        Configure the meta-optimizer.
        """
        return torch.optim.Adam(self.module.parameters(), lr=self.meta_lr)

    def training_step(self, batch, batch_idx):
        """
        Perform a MAML training step (Inner + Outer loops).
        """
        # Batch: Tuple of (Support Set, Query Set) for a task
        # Assuming batch is a list of tasks, but for simplicity handling one task per step here
        # or batch dim is task dim.
        
        support_data, query_data = batch
        
        # Inner Loop (Adaptation)
        with higher.innerloop_ctx(self.module, self.module.configure_optimizers(), copy_initial_weights=False) as (fmodel, diffopt):
            for _ in range(self.num_inner_steps):
                # Calculate Support Loss
                # This requires the inner module's training_step to be callable/compatible
                loss = fmodel.training_step(support_data, 0) # batch_idx=0
                diffopt.step(loss)
            
            # Outer Loop (Meta-Update)
            # Evaluate on Query Set using adapted weights (fmodel)
            meta_loss = fmodel.validation_step(query_data, 0) # Using Val logic for query split
            
            self.log('train/meta_loss', meta_loss)
            return meta_loss
