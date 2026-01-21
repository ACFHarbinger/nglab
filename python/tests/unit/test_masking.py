import unittest

import torch

from python.src.utils.functions.masking import ProbMask, TriangularCausalMask


class TestMasking(unittest.TestCase):
    def test_triangular_causal_mask_cpu(self):
        mask = TriangularCausalMask(B=2, L=4, device="cpu")
        self.assertEqual(mask.mask.shape, (2, 1, 4, 4))
        self.assertEqual(mask.mask.dtype, torch.bool)
        # Verify it's upper triangular (diagonal=1)
        self.assertFalse(mask.mask[0, 0, 0, 0].item())
        self.assertTrue(mask.mask[0, 0, 0, 1].item())

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_triangular_causal_mask_cuda(self):
        mask = TriangularCausalMask(B=2, L=4, device="cuda")
        self.assertEqual(mask.mask.device.type, "cuda")

    def test_prob_mask(self):
        B, H, L = 2, 4, 6
        index = torch.randint(0, L, (B, H, 3))
        scores = torch.randn(B, H, 3, 10)
        mask = ProbMask(B, H, L, index, scores, device="cpu")
        self.assertEqual(mask.mask.shape, scores.shape)
        self.assertEqual(mask.mask.dtype, torch.bool)
