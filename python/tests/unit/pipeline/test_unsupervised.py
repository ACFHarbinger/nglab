import unittest

import torch

from python.src.pipeline.core.lightning.self_supervised import SelfSupervisedModule
from python.src.pipeline.core.lightning.unsupervised_learning import UnsupervisedModule


class TestUnsupervised(unittest.TestCase):
    def setUp(self):
        self.backbone = torch.nn.Linear(10, 10)
        self.cfg = {"input_dim": 10, "hidden_dim": 10}

    def test_self_supervised(self):
        module = SelfSupervisedModule(self.backbone, self.cfg)
        # Dummy batch
        batch = torch.randn(5, 10)
        loss = module.training_step(batch, 0)
        self.assertIsInstance(loss, torch.Tensor)

    def test_unsupervised(self):
        module = UnsupervisedModule(self.backbone, self.cfg)
        batch = torch.randn(5, 10)
        loss = module.training_step(batch, 0)
        self.assertIsInstance(loss, torch.Tensor)


if __name__ == "__main__":
    unittest.main()
