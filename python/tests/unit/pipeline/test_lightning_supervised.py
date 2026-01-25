import json
import unittest
from unittest.mock import ANY, MagicMock, patch

import torch
from torch import nn

from python.src.pipeline.core.lightning.supervised_learning import (
    ProgressCallback,
    SLLightningModule,
)


class TestLightningSupervised(unittest.TestCase):
    def setUp(self):
        self.cfg = {"hidden_dim": 64, "output_dim": 1, "learning_rate": 0.001}
        self.backbone = nn.Linear(32, 64)
        self.module = SLLightningModule(self.backbone, self.cfg)

    def test_init(self):
        self.assertEqual(self.module.backbone, self.backbone)
        self.assertIsInstance(self.module.head, nn.Linear)
        self.assertEqual(self.module.head.out_features, 1)

    def test_forward(self):
        x = torch.randn(8, 32)
        out = self.module(x)
        self.assertEqual(out.shape, (8, 1))

    def test_training_step(self):
        # Mock batch as dict
        batch = {"observation": torch.randn(8, 32), "target": torch.randn(8, 1)}
        # We need to mock .log to avoid Lightning module error if not in trainer
        self.module.log = MagicMock()

        loss = self.module.training_step(batch, 0)
        self.assertIsInstance(loss, torch.Tensor)
        self.module.log.assert_called_with("train/sl_loss", ANY)

    def test_training_step_tuple(self):
        # Mock batch as tuple
        batch = (torch.randn(8, 32), torch.randn(8, 1))
        self.module.log = MagicMock()

        loss = self.module.training_step(batch, 0)
        self.assertIsInstance(loss, torch.Tensor)

    def test_validation_step(self):
        batch = {"observation": torch.randn(8, 32), "target": torch.randn(8, 1)}
        self.module.log = MagicMock()

        loss = self.module.validation_step(batch, 0)
        self.assertIsInstance(loss, torch.Tensor)
        self.module.log.assert_called_with("val/sl_loss", ANY)

    def test_progress_callback(self):
        callback = ProgressCallback(total_epochs=10)
        with patch("builtins.print") as mock_print:
            callback.on_epoch_end(epoch=0, train_loss=0.5, val_loss=0.6)
            mock_print.assert_called()
            args, _ = mock_print.call_args
            progress = json.loads(args[0])
            self.assertEqual(progress["type"], "progress")
            self.assertEqual(progress["epoch"], 1)
            self.assertEqual(progress["percent"], 10.0)
