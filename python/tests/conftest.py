import sys
from pathlib import Path

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
