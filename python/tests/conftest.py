import gzip
import os
import sys
import warnings
from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytest_plugins = [
    "python.tests.fixtures.arg_fixtures",
    "python.tests.fixtures.config_fixtures",
    "python.tests.fixtures.deep_fixtures",
    "python.tests.fixtures.environment_fixtures",
    "python.tests.fixtures.hpo_fixtures",
    "python.tests.fixtures.mac_fixtures",
    "python.tests.fixtures.model_fixtures",
    "python.tests.fixtures.nglab_fixtures",
    "python.tests.fixtures.pipeline_fixtures",
    "python.tests.fixtures.policy_fixtures",
    "python.tests.fixtures.regression_fixtures",
    "python.tests.fixtures.tensor_fixtures",
    "python.tests.fixtures.utils_fixtures",
]

# Filter stubborn warnings that pyproject.toml misses
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
warnings.filterwarnings("ignore", message=".*ModuleAvailableCache.*")
warnings.filterwarnings("ignore", message=".*torch_geometric.distributed.*")

# Disable OpenTelemetry during tests to prevent Jaeger connection warnings
os.environ["OTEL_SDK_DISABLED"] = "true"

# Global mock for zstandard as it's an optional dependency often missing in CI
mock_zstd = MagicMock()
mock_zstd.__name__ = "zstandard"
mock_zstd.__version__ = "0.15.2"
mock_zstd.ZstdCompressor.return_value.compress.side_effect = gzip.compress
mock_zstd.ZstdDecompressor.return_value.decompress.side_effect = gzip.decompress
sys.modules["zstandard"] = mock_zstd

# The project root is THREE levels up from conftest.py:
# conftest.py -> tests -> python -> nglab (Project Root)
# /home/pkhunter/Repositories/nglab/python/tests/conftest.py
# .parent -> python/tests
# .parent.parent -> python
# .parent.parent.parent -> nglab (Project Root)
project_root = Path(__file__).resolve().parent.parent.parent

# Add the project root to sys.path.
# This allows 'import python.src...' to resolve correctly.
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


# GPU test support
@pytest.fixture(scope="session")
def cuda_available():
    """Check if CUDA is available for GPU tests."""
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


@pytest.fixture(scope="session")
def device(cuda_available):
    """Return the primary device (cuda:0 or cpu)."""
    import torch

    return torch.device("cuda:0" if cuda_available else "cpu")


@pytest.fixture(autouse=True)
def clean_gpu_cache(cuda_available):
    """Automatically clear GPU cache after each test to prevent side effects."""
    yield
    if cuda_available:
        import torch

        torch.cuda.empty_cache()


def pytest_configure(config):
    """Register the 'gpu' marker."""
    config.addinivalue_line("markers", "gpu: mark tests that require a GPU")


def pytest_collection_modifyitems(config, items):
    """Automatically skip GPU tests when CUDA is not available."""
    try:
        import torch

        has_cuda = torch.cuda.is_available()
    except ImportError:
        has_cuda = False

    skip_gpu = pytest.mark.skip(reason="GPU/CUDA not available")

    for item in items:
        if "gpu" in item.keywords and not has_cuda:
            item.add_marker(skip_gpu)
