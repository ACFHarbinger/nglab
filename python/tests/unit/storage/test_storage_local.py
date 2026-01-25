from pathlib import Path

import pytest

from python.src.storage.base import StorageConfig
from python.src.storage.local import LocalStorage


@pytest.fixture
def local_storage(tmp_path):
    config = StorageConfig(
        storage_type="local",
        local_path=str(tmp_path / "models"),
        versioning=True,
        max_versions=3,
    )
    return LocalStorage(config)


class TestLocalStorage:
    def test_init_creates_directory(self, tmp_path):
        config = StorageConfig(
            storage_type="local",
            local_path=str(tmp_path / "new_models"),
        )
        LocalStorage(config)
        assert (tmp_path / "new_models").exists()

    def test_save_and_load_model(self, local_storage):
        model_data = b"test_model_data"
        name = "test_model"

        path = local_storage.save(model_data, name)
        assert Path(path).exists()

        loaded_data = local_storage.load(name)
        assert loaded_data == model_data

    def test_save_generated_version(self, local_storage):
        model_data = b"data"
        name = "auto_version_model"

        local_storage.save(model_data, name)

        versions = local_storage.list_versions(name)
        assert len(versions) == 1
        # Generated version should look like timestamp
        assert "_" in versions[0]

    def test_save_explicit_version(self, local_storage):
        model_data = b"data"
        name = "explicit_version_model"
        version = "v1.0.0"

        local_storage.save(model_data, name, version=version)

        versions = local_storage.list_versions(name)
        assert version in versions

    def test_metadata_saved(self, local_storage):
        model_data = b"data"
        name = "meta_model"
        extra_meta = {"extra": {"accuracy": 0.95}}

        local_storage.save(model_data, name, metadata=extra_meta)

        meta = local_storage.get_metadata(name)
        assert meta.name == name
        assert meta.size_bytes == len(model_data)
        assert meta.extra.get("accuracy") == 0.95

    def test_load_latest(self, local_storage):
        name = "latest_test"
        local_storage.save(b"v1", name, version="v1")
        local_storage.save(b"v2", name, version="v2")

        data = local_storage.load(name)
        assert data == b"v2"

        meta = local_storage.get_metadata(name)
        assert meta.version == "v2"

    def test_load_specific_version(self, local_storage):
        name = "version_test"
        local_storage.save(b"v1", name, version="v1")
        local_storage.save(b"v2", name, version="v2")

        data = local_storage.load(name, version="v1")
        assert data == b"v1"

    def test_exists(self, local_storage):
        name = "exists_test"
        local_storage.save(b"data", name, version="v1")

        assert local_storage.exists(name)
        assert local_storage.exists(name, "v1")
        assert not local_storage.exists(name, "v2")
        assert not local_storage.exists("non_existent")

    def test_delete_version(self, local_storage):
        name = "delete_version_test"
        local_storage.save(b"v1", name, version="v1")
        local_storage.save(b"v2", name, version="v2")

        assert local_storage.delete(name, "v1")
        assert not local_storage.exists(name, "v1")
        assert local_storage.exists(name, "v2")

    def test_delete_model(self, local_storage):
        name = "delete_model_test"
        local_storage.save(b"v1", name, version="v1")

        assert local_storage.delete(name)
        assert not local_storage.exists(name)
        assert not (local_storage.base_path / name).exists()

    def test_list_models(self, local_storage):
        names = ["m1", "m2", "m3"]
        for n in names:
            local_storage.save(b"data", n)

        listed = local_storage.list_models()
        assert sorted(listed) == sorted(names)

    def test_cleanup_old_versions(self, local_storage):
        # max_versions=3 from fixture
        name = "cleanup_test"
        versions = ["v1", "v2", "v3", "v4", "v5"]

        for v in versions:
            local_storage.save(b"data", name, version=v)

        remaining = local_storage.list_versions(name)
        assert len(remaining) == 3
        # Should keep newest: v5, v4, v3
        assert set(remaining) == {"v5", "v4", "v3"}
        assert not local_storage.exists(name, "v1")
        assert not local_storage.exists(name, "v2")

    def test_load_not_found(self, local_storage):
        with pytest.raises(FileNotFoundError):
            local_storage.load("non_existent")

        with pytest.raises(FileNotFoundError):
            local_storage.get_metadata("non_existent")

    def test_version_not_found(self, local_storage):
        name = "missing_ver"
        local_storage.save(b"data", name, version="v1")

        with pytest.raises(FileNotFoundError):
            local_storage.load(name, version="v999")
