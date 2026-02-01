from __future__ import annotations

import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def temp_output_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)

@pytest.fixture
def mock_logger() -> Generator[Any, None, None]:
    """Mock logger for testing logging output."""
    from unittest.mock import MagicMock
    yield MagicMock()

__all__ = ["mock_logger", "temp_output_dir"]
