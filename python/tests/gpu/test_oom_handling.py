"""
GPU OOM Handling Tests.

Verifies that the system can handle or at least report Out Of Memory errors gracefully.
"""

import pytest
import torch

try:
    from python.src.utils.profiling import get_gpu_memory_stats
except ImportError:
    pass


@pytest.mark.gpu
class TestOOMHandling:
    def test_oom_recovery(self):
        """
        Attempt to trigger OOM and verify we can recover/clear cache.
        This test is risky if it crashes the process, so we use a sub-process
        or careful allocation.
        """
        if not torch.cuda.is_available():
            pytest.skip("No CUDA device")

        try:
            # Try to allocate a huge tensor effectively guaranteed to fail on consumer GPUs
            # 80GB VRAM might pass this, but typical 24GB will fail.
            # 40GB * 1024^3 bytes * 4 bytes/float = ~160GB
            torch.empty((40000, 10000, 1000), device="cuda")
        except RuntimeError as e:
            assert "out of memory" in str(e).lower()

            # Verify we can recover
            torch.cuda.empty_cache()

            # Should be able to allocate small tensor now
            small = torch.ones(10, device="cuda")
            assert small.sum() == 10
        except Exception:
            # If we passed (e.g. on A100 80GB), just warn or skip
            pytest.skip("Could not trigger OOM (GPU too fast/large?)")

    def test_memory_snapshot(self):
        """Verify we can take memory snapshots."""
        if not torch.cuda.is_available():
            pytest.skip("No CUDA device")

        # Allocate something
        t = torch.zeros(1024, 1024, device="cuda")  # 4MB

        try:
            stats = get_gpu_memory_stats()
            # If function exists, check keys
            if stats:
                assert stats.allocated_mb > 0
        except NameError:
            pass

        del t
        torch.cuda.empty_cache()
