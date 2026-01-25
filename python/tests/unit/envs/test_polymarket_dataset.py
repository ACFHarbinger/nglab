import json

import numpy as np
import pandas as pd
import pytest
import torch

from python.src.data.polymarket_dataset import PolymarketDataset


@pytest.fixture
def dataset_setup(tmp_path):
    dataset_dir = tmp_path / "polymarket"
    dataset_dir.mkdir()

    metadata = [{"id": 0, "filename": "c1.csv"}, {"id": 1, "filename": "c2.csv"}]

    metadata_path = dataset_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)

    # Create CSVs
    df1 = pd.DataFrame(
        {
            "Timestamp (UTC)": pd.date_range("2023-01-01", periods=10, freq="H"),
            "Price": np.linspace(0.1, 0.5, 10),
        }
    )
    df1.to_csv(dataset_dir / "c1.csv", index=False)

    df2 = pd.DataFrame(
        {
            "Timestamp (UTC)": pd.date_range("2023-01-01", periods=10, freq="H"),
            "Price": np.linspace(0.5, 0.9, 10),
        }
    )
    df2.to_csv(dataset_dir / "c2.csv", index=False)

    return str(dataset_dir)


def test_polymarket_dataset_init(dataset_setup):
    dataset = PolymarketDataset(
        name="test", dataset_dir=dataset_setup, seq_len=5, pred_len=2
    )

    assert dataset.name == "test"
    assert dataset.seq_len == 5
    assert dataset.pred_len == 2
    assert len(dataset) == 10 - 5 - 2 + 1  # 4


def test_polymarket_dataset_getitem(dataset_setup):
    dataset = PolymarketDataset(
        name="test", dataset_dir=dataset_setup, seq_len=5, pred_len=2
    )

    sample = dataset[0]
    assert "Price" in sample
    assert "Labels" in sample
    assert sample["Price"].shape == (5, 2)
    assert sample["Labels"].shape == (2, 2)

    # Check values
    # c1 prices: 0.1, 0.144..., c2 prices: 0.5, 0.544...
    assert torch.allclose(sample["Price"][0], torch.tensor([0.1, 0.5]), atol=1e-5)


def test_polymarket_dataset_download_error(dataset_setup):
    with pytest.raises(NotImplementedError):
        PolymarketDataset("test", dataset_setup, 5, 2, download=True)


def test_polymarket_dataset_too_short(dataset_setup):
    with pytest.raises(ValueError, match="too short"):
        PolymarketDataset("test", dataset_setup, 10, 5)


def test_polymarket_dataset_no_valid_data(tmp_path):
    dataset_dir = tmp_path / "empty"
    dataset_dir.mkdir()
    metadata_path = dataset_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump([], f)

    with pytest.raises(ValueError, match="No valid data frames"):
        PolymarketDataset("test", str(dataset_dir), 5, 2)


def test_polymarket_dataset_transform(dataset_setup):
    def transform(data):
        data["Price"] = data["Price"] * 2
        return data

    dataset = PolymarketDataset("test", dataset_setup, 5, 2, transform=transform)
    sample = dataset[0]
    assert torch.allclose(sample["Price"][0], torch.tensor([0.2, 1.0]), atol=1e-5)
