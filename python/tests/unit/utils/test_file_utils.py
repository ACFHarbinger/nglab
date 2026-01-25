"""
Tests for file utility functions.
"""

import json
import threading
from unittest.mock import MagicMock

import pytest

from python.src.utils.io.file_utils import compose_dirpath, read_json


class TestComposeDirpath:
    """Tests for the compose_dirpath decorator.

    Note: The compose_dirpath decorator checks os.path.isfile(path) to determine
    whether to create the dirname or the full path. If the file doesn't exist yet,
    isfile returns False and it creates the full path as a directory.
    This means it's designed to be used with dir_path parameters, not file paths.
    """

    def test_creates_directory_for_dir_path(self, tmp_path):
        """Test that decorator creates directory when given a directory path."""
        target_dir = tmp_path / "new_subdir"

        @compose_dirpath
        def use_dir(dir_path):
            return dir_path

        assert not target_dir.exists()

        result = use_dir(str(target_dir))

        assert target_dir.exists()
        assert result == str(target_dir)

    def test_handles_nested_directories(self, tmp_path):
        """Test creating deeply nested directories."""
        nested_dir = tmp_path / "a" / "b" / "c"

        @compose_dirpath
        def deep_create(path):
            return path

        result = deep_create(str(nested_dir))

        assert nested_dir.exists()
        assert result == str(nested_dir)

    def test_handles_existing_directory(self, tmp_path):
        """Test that decorator works when directory already exists."""
        existing_dir = tmp_path / "existing"
        existing_dir.mkdir()

        @compose_dirpath
        def use_existing(dir_path):
            return "success"

        result = use_existing(str(existing_dir))

        assert result == "success"
        assert existing_dir.exists()

    def test_creates_parent_for_existing_file(self, tmp_path):
        """Test that decorator creates parent dir when path is an existing file."""
        # Create a file first
        target_dir = tmp_path / "has_file"
        target_dir.mkdir()
        target_file = target_dir / "existing.json"
        target_file.write_text("{}")

        @compose_dirpath
        def read_existing(path):
            with open(path) as f:
                return f.read()

        # Now the file exists, so isfile returns True
        # Decorator will call dirname (which is target_dir) and makedirs it
        result = read_existing(str(target_file))

        assert result == "{}"

    def test_handles_kwargs_json_path(self, tmp_path):
        """Test that decorator handles json_path keyword argument."""
        target_dir = tmp_path / "json_dir"
        target_file = target_dir / "data.json"

        @compose_dirpath
        def save_json(json_path=None):
            return json_path

        result = save_json(json_path=str(target_file))

        assert target_dir.exists()
        assert result == str(target_file)

    def test_handles_kwargs_dir_path(self, tmp_path):
        """Test that decorator handles dir_path keyword argument."""
        target_dir = tmp_path / "dir_path_test"

        @compose_dirpath
        def use_dir(dir_path=None):
            return dir_path

        result = use_dir(dir_path=str(target_dir))

        assert target_dir.exists()
        assert result == str(target_dir)

    def test_no_path_provided(self):
        """Test that decorator handles case when no path is provided."""

        @compose_dirpath
        def no_path_func():
            return "no path"

        # Should not raise error
        result = no_path_func()
        assert result == "no path"

    def test_preserves_function_behavior(self, tmp_path):
        """Test that decorator preserves original function behavior."""
        target_dir = tmp_path / "preserve_test"
        target_file = target_dir / "test.txt"

        @compose_dirpath
        def complex_function(path, *args, **kwargs):
            return {
                "path": path,
                "args": args,
                "kwargs": kwargs,
            }

        result = complex_function(str(target_file), "arg1", "arg2", key="value")

        assert result["path"] == str(target_file)
        assert result["args"] == ("arg1", "arg2")
        assert result["kwargs"] == {"key": "value"}


class TestReadJson:
    """Tests for the read_json function."""

    def test_read_simple_json(self, tmp_path):
        """Test reading a simple JSON file."""
        json_path = tmp_path / "simple.json"
        data = {"key": "value", "number": 42}

        with open(json_path, "w") as f:
            json.dump(data, f)

        result = read_json(str(json_path))

        assert result == data
        assert result["key"] == "value"
        assert result["number"] == 42

    def test_read_nested_json(self, tmp_path):
        """Test reading nested JSON data."""
        json_path = tmp_path / "nested.json"
        data = {
            "level1": {"level2": {"level3": ["a", "b", "c"]}},
            "array": [1, 2, 3],
        }

        with open(json_path, "w") as f:
            json.dump(data, f)

        result = read_json(str(json_path))

        assert result == data
        assert result["level1"]["level2"]["level3"] == ["a", "b", "c"]

    def test_read_json_array(self, tmp_path):
        """Test reading JSON array."""
        json_path = tmp_path / "array.json"
        data = [{"id": 1}, {"id": 2}, {"id": 3}]

        with open(json_path, "w") as f:
            json.dump(data, f)

        result = read_json(str(json_path))

        assert result == data
        assert len(result) == 3

    def test_read_json_with_lock(self, tmp_path):
        """Test reading JSON with a threading lock."""
        json_path = tmp_path / "locked.json"
        data = {"thread_safe": True}

        with open(json_path, "w") as f:
            json.dump(data, f)

        lock = threading.Lock()
        result = read_json(str(json_path), lock=lock)

        assert result == data
        # Lock should be released after reading
        assert not lock.locked()

    def test_lock_acquired_and_released(self, tmp_path):
        """Test that lock is properly acquired and released."""
        json_path = tmp_path / "lock_test.json"
        data = {"test": "data"}

        with open(json_path, "w") as f:
            json.dump(data, f)

        lock = MagicMock()

        read_json(str(json_path), lock=lock)

        lock.acquire.assert_called_once()
        lock.release.assert_called_once()

    def test_lock_released_on_error(self, tmp_path):
        """Test that lock is released even if an error occurs."""
        json_path = tmp_path / "invalid.json"

        # Write invalid JSON
        with open(json_path, "w") as f:
            f.write("not valid json {{{")

        lock = MagicMock()

        with pytest.raises(json.JSONDecodeError):
            read_json(str(json_path), lock=lock)

        # Lock should still be released
        lock.release.assert_called_once()

    def test_file_not_found(self, tmp_path):
        """Test error when file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            read_json(str(tmp_path / "nonexistent.json"))

    def test_invalid_json(self, tmp_path):
        """Test error when JSON is invalid."""
        json_path = tmp_path / "invalid.json"

        with open(json_path, "w") as f:
            f.write("{ invalid json }")

        with pytest.raises(json.JSONDecodeError):
            read_json(str(json_path))

    def test_empty_json_object(self, tmp_path):
        """Test reading empty JSON object."""
        json_path = tmp_path / "empty.json"

        with open(json_path, "w") as f:
            f.write("{}")

        result = read_json(str(json_path))

        assert result == {}

    def test_empty_json_array(self, tmp_path):
        """Test reading empty JSON array."""
        json_path = tmp_path / "empty_array.json"

        with open(json_path, "w") as f:
            f.write("[]")

        result = read_json(str(json_path))

        assert result == []

    def test_json_with_unicode(self, tmp_path):
        """Test reading JSON with unicode characters."""
        json_path = tmp_path / "unicode.json"
        data = {"message": "Hello 世界 🌍", "emoji": "🎉"}

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

        result = read_json(str(json_path))

        assert result == data
        assert "世界" in result["message"]

    def test_large_json(self, tmp_path):
        """Test reading large JSON file."""
        json_path = tmp_path / "large.json"
        data = {f"key_{i}": f"value_{i}" * 100 for i in range(1000)}

        with open(json_path, "w") as f:
            json.dump(data, f)

        result = read_json(str(json_path))

        assert len(result) == 1000
        assert result["key_0"].startswith("value_0")


class TestConcurrentAccess:
    """Tests for concurrent file access scenarios."""

    def test_concurrent_reads_with_lock(self, tmp_path):
        """Test multiple threads can read safely with lock."""
        json_path = tmp_path / "concurrent.json"
        data = {"counter": 42}

        with open(json_path, "w") as f:
            json.dump(data, f)

        lock = threading.Lock()
        results = []
        errors = []

        def reader():
            try:
                result = read_json(str(json_path), lock=lock)
                results.append(result)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=reader) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 10
        assert all(r == data for r in results)

    def test_lock_not_required_for_reads(self, tmp_path):
        """Test that lock is optional and None works."""
        json_path = tmp_path / "no_lock.json"
        data = {"no_lock": True}

        with open(json_path, "w") as f:
            json.dump(data, f)

        # Should work without lock
        result = read_json(str(json_path), lock=None)
        assert result == data

        # Should work with default (no lock argument)
        result = read_json(str(json_path))
        assert result == data
