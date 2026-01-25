from __future__ import annotations

import logging
import os
from typing import Any
import psutil
from pytorch_lightning.callbacks import Callback
import pytorch_lightning as pl

logger = logging.getLogger(__name__)

class MemoryTrackingCallback(Callback):
    """
    Log memory usage at the end of each epoch.
    """
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        rss_mb = mem_info.rss / (1024 * 1024)
        vms_mb = mem_info.vms / (1024 * 1024)
        
        pl_module.log("memory/rss_mb", rss_mb, prog_bar=False)
        pl_module.log("memory/vms_mb", vms_mb, prog_bar=False)
        
        logger.info(f"Epoch {trainer.current_epoch} Memory Usage: RSS={rss_mb:.2f}MB, VMS={vms_mb:.2f}MB")

class PerformanceLoggingCallback(Callback):
    """
    Log training performance metrics.
    """
    def on_train_batch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: Any, batch: Any, batch_idx: int
    ) -> None:
        # Pytorch Lightning usually handles time/step logs, but we can add custom ones
        pass

__all__ = ["MemoryTrackingCallback", "PerformanceLoggingCallback"]
