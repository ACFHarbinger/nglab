import unittest
from unittest.mock import MagicMock

import torch

from python.src.pipeline.core.lightning.semi_supervised import SemiSupervisedModule


class TestSemiSupervisedModule(unittest.TestCase):
    def setUp(self):
        self.cfg = {
            "hidden_dim": 64,
            "num_classes": 3,
            "threshold": 0.9,
            "lambda_u": 2.0,
            "learning_rate": 0.001,
        }
        self.backbone = torch.nn.Linear(32, 64)
        self.module = SemiSupervisedModule(self.backbone, self.cfg)

    def test_init(self):
        self.assertEqual(self.module.threshold, 0.9)
        self.assertEqual(self.module.lambda_u, 2.0)
        self.assertIsInstance(self.module.head, torch.nn.Linear)
        self.assertEqual(self.module.head.out_features, 3)

    def test_forward(self):
        x = torch.randn(8, 32)
        output = self.module(x)
        self.assertEqual(output.shape, (8, 3))

    def test_training_step_labeled_only(self):
        batch = {
            "labeled": (torch.randn(8, 32), torch.randint(0, 3, (8,))),
            "unlabeled": None,
        }
        self.module.log = MagicMock()

        # Move module to CPU explicitly
        self.module.cpu()
        loss = self.module.training_step(batch, 0)

        self.assertIsInstance(loss, torch.Tensor)
        self.assertTrue(loss.item() > 0)

    def test_training_step_with_unlabeled(self):
        batch = {
            "labeled": (torch.randn(4, 32), torch.randint(0, 3, (4,))),
            "unlabeled": torch.randn(8, 32),
        }
        self.module.log = MagicMock()

        # Move module to CPU explicitly
        self.module.cpu()
        loss = self.module.training_step(batch, 0)

        self.assertIsInstance(loss, torch.Tensor)
        self.assertTrue(loss.item() > 0)

    def test_training_step_unlabeled_only(self):
        batch = {"labeled": (None, None), "unlabeled": torch.randn(8, 32)}
        self.module.log = MagicMock()

        # Move module to CPU explicitly
        self.module.cpu()
        loss = self.module.training_step(batch, 0)

        self.assertIsInstance(loss, torch.Tensor)

    def test_training_step_invalid_batch(self):
        # Test that non-dict batch raises error
        batch = (torch.randn(8, 32), torch.randint(0, 3, (8,)))

        with self.assertRaises(ValueError):
            self.module.training_step(batch, 0)
