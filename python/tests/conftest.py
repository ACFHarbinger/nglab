import sys
from pathlib import Path

import pytest

# The project root is THREE levels up from conftest.py:
# conftest.py -> tests -> python -> nglab (Project Root)
# /home/pkhunter/Repositories/nglab/python/tests/conftest.py
# .parent -> python/tests
# .parent.parent -> python
# .parent.parent.parent -> nglab (Project Root)
project_root = Path(__file__).resolve().parent.parent.parent

# Add the project root to sys.path.
# This allows 'import python.src...' to resolve correctly.
sys.path.insert(0, str(project_root))

# Load modular fixtures
pytest_plugins = [
    "python.tests.fixtures.deep_fixtures",
    "python.tests.fixtures.mac_fixtures",
    "python.tests.fixtures.regression_fixtures",
]


# GPU test support
@pytest.fixture(scope="session")
def cuda_available():
    """Check if CUDA is available for GPU tests."""
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


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

