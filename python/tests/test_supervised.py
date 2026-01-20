import unittest

import torch

from python.src.pipeline.core.lightning.supervised_learning import SLLightningModule


class TestSupervised(unittest.TestCase):
    def setUp(self):
        self.backbone = torch.nn.Linear(10, 10)
        self.cfg = {"output_dim": 1, "hidden_dim": 10}

    def test_supervised_step(self):
        module = SLLightningModule(self.backbone, self.cfg)
        batch = {"observation": torch.randn(5, 10), "target": torch.randn(5, 1)}
        loss = module.training_step(batch, 0)
        self.assertIsInstance(loss, torch.Tensor)


if __name__ == "__main__":
    unittest.main()
