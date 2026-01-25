import sys
import warnings
from pathlib import Path

# Filter stubborn warnings that occur early in import time
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
warnings.filterwarnings("ignore", message=".*ModuleAvailableCache.*")
warnings.filterwarnings("ignore", message=".*torch_geometric.distributed.*")
warnings.filterwarnings("ignore", category=ImportWarning)
try:
    from sklearn.exceptions import LinAlgWarning

    warnings.filterwarnings("ignore", category=LinAlgWarning)
except ImportError:
    pass

# Filter unawaited coroutine warnings if they persist (though we aim to fix them)
warnings.filterwarnings(
    "ignore", message=".*coroutine.*was never awaited.*", category=RuntimeWarning
)

# Add project root to path
# /home/pkhunter/Repositories/nglab/conftest.py -> .parent is /home/pkhunter/Repositories/nglab
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

pytest_plugins = [
    "python.tests.fixtures.deep_fixtures",
    "python.tests.fixtures.mac_fixtures",
    "python.tests.fixtures.regression_fixtures",
    "python.tests.fixtures.hpo_fixtures",
    "python.tests.fixtures.model_fixtures",
    "python.tests.fixtures.environment_fixtures",
    "python.tests.fixtures.nglab_fixtures",
]
