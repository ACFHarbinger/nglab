import unittest

import torch

from python.src.pipeline.core.lightning.base import BaseModule


class ConcreteModule(BaseModule):
    """Concrete implementation of BaseModule for testing."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.layer = torch.nn.Linear(10, 1)

    def training_step(self, batch, batch_idx):
        return torch.tensor(0.5)

    def validation_step(self, batch, batch_idx):
        return torch.tensor(0.3)


class TestBaseModule(unittest.TestCase):
    def test_init_default_lr(self):
        cfg = {}
        module = ConcreteModule(cfg)
        self.assertEqual(module.learning_rate, 1e-3)
        self.assertEqual(module.cfg, cfg)

    def test_init_custom_lr(self):
        cfg = {"learning_rate": 0.01}
        module = ConcreteModule(cfg)
        self.assertEqual(module.learning_rate, 0.01)

    def test_configure_optimizers(self):
        cfg = {"learning_rate": 0.001}
        module = ConcreteModule(cfg)
        optimizer = module.configure_optimizers()
        self.assertIsInstance(optimizer, torch.optim.Adam)
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.001)

    def test_training_step_not_implemented(self):
        cfg = {}
        module = BaseModule(cfg)
        with self.assertRaises(NotImplementedError):
            module.training_step({}, 0)

    def test_validation_step_not_implemented(self):
        cfg = {}
        module = BaseModule(cfg)
        with self.assertRaises(NotImplementedError):
            module.validation_step({}, 0)
