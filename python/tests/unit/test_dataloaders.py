"""Unit tests for data modules."""

import numpy as np
import pandas as pd
import pytest
import torch

from python.src.data.data_utils import df_to_torch, read_csv, read_json
from python.src.data.dataloaders import (
    FinancialDataset,
    StreamingDataset,
    create_dataloader,
)


class TestDataUtils:
    """Tests for data utility functions."""

    def test_read_json(self, tmp_path):
        """Test JSON reading."""
        json_file = tmp_path / "test.json"
        json_file.write_text('{"key": "value", "number": 42}')

        data = read_json(str(json_file))
        assert data["key"] == "value"
        assert data["number"] == 42

    def test_read_csv(self, tmp_path):
        """Test CSV reading."""
        csv_file = tmp_path / "test.csv"
        df_data = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df_data.to_csv(csv_file, index=False)

        df = read_csv(str(csv_file))
        assert df is not None
        assert len(df) == 3
        assert list(df.columns) == ["a", "b"]

    def test_read_csv_nonexistent(self):
        """Test reading non-existent CSV returns None."""
        df = read_csv("/nonexistent/file.csv")
        assert df is None

    def test_df_to_torch(self):
        """Test DataFrame to torch tensor conversion."""
        df = pd.DataFrame(
            {
                "feature1": [1.0, 2.0, 3.0],
                "feature2": [4.0, 5.0, 6.0],
                "date": ["2021-01-01", "2021-01-02", "2021-01-03"],
            }
        )

        torch_dict = df_to_torch(df, key_avoid="date")

        assert "feature1" in torch_dict
        assert "feature2" in torch_dict
        assert "date" not in torch_dict

        assert isinstance(torch_dict["feature1"], torch.Tensor)
        assert torch.equal(torch_dict["feature1"], torch.tensor([1.0, 2.0, 3.0]))

    def test_df_to_torch_fill_nan(self):
        """Test NaN filling in DataFrame to torch conversion."""
        df = pd.DataFrame({"values": [1.0, np.nan, 3.0]})

        torch_dict = df_to_torch(df, key_avoid="none", fill_nan=0.0)

        assert torch.equal(torch_dict["values"], torch.tensor([1.0, 0.0, 3.0]))


class TestFinancialDataset:
    """Tests for FinancialDataset."""

    @pytest.fixture
    def sample_csv(self, tmp_path):
        """Create a sample financial CSV file."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=100),
                "price": np.random.randn(100).cumsum() + 100,
                "volume": np.random.randint(1000, 10000, 100),
                "returns": np.random.randn(100) * 0.01,
            }
        )
        csv_path = tmp_path / "financial_data.csv"
        df.to_csv(csv_path, index=False)
        return str(csv_path)

    def test_init_basic(self, sample_csv):
        """Test basic initialization."""
        dataset = FinancialDataset(
            csv_path=sample_csv, target_column="price", seq_len=10, pred_len=1
        )

        assert dataset.seq_len == 10
        assert dataset.pred_len == 1
        assert len(dataset) > 0

    def test_getitem(self, sample_csv):
        """Test getting an item from dataset."""
        dataset = FinancialDataset(
            csv_path=sample_csv,
            target_column="price",
            seq_len=10,
            pred_len=1,
            normalize=None,
        )

        item = dataset[0]
        # TimeSeriesDataset uses 'observation' and 'target' keys
        assert "observation" in item
        assert "target" in item

        # Shapes should match
        assert item["observation"].shape[0] == 10
        assert item["target"].shape[0] == 1

    def test_train_val_split(self, sample_csv):
        """Test train/val splitting."""
        train_dataset = FinancialDataset(
            csv_path=sample_csv,
            target_column="price",
            seq_len=10,
            pred_len=1,
            train=True,
            train_ratio=0.7,
        )

        val_dataset = FinancialDataset(
            csv_path=sample_csv,
            target_column="price",
            seq_len=10,
            pred_len=1,
            train=False,
            train_ratio=0.7,
            stats={
                "min": train_dataset.raw_min,
                "max": train_dataset.raw_max,
                "mean": train_dataset.raw_mean,
                "std": train_dataset.raw_std,
            },
        )

        # Train should be larger
        assert len(train_dataset) > len(val_dataset)


class TestCreateDataloader:
    """Tests for create_dataloader factory function."""

    @pytest.fixture
    def sample_csv(self, tmp_path):
        """Create sample CSV for dataloader tests."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=200),
                "price": np.random.randn(200).cumsum() + 100,
                "volume": np.random.randint(1000, 10000, 200),
            }
        )
        csv_path = tmp_path / "data.csv"
        df.to_csv(csv_path, index=False)
        return str(csv_path)

    def test_create_dataloader_basic(self, sample_csv):
        """Test basic dataloader creation."""
        train_loader, val_loader, test_loader = create_dataloader(
            data_path=sample_csv,
            target_column="price",
            batch_size=16,
            seq_len=10,
            pred_len=1,
            num_workers=0,  # Avoid multiprocessing in tests
        )

        assert train_loader is not None
        assert val_loader is not None
        assert test_loader is not None

    def test_create_dataloader_ratios(self, sample_csv):
        """Test dataloader with custom ratios."""
        train_loader, val_loader, test_loader = create_dataloader(
            data_path=sample_csv,
            target_column="price",
            batch_size=8,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
            num_workers=0,
        )

        # Verify all loaders are created
        assert len(train_loader) > 0
        assert len(val_loader) > 0
        assert len(test_loader) > 0

    def test_create_dataloader_invalid_ratios(self, sample_csv):
        """Test that invalid ratios raise an error."""
        with pytest.raises(AssertionError, match=r"must sum to 1\.0"):
            create_dataloader(
                data_path=sample_csv,
                target_column="price",
                train_ratio=0.5,
                val_ratio=0.3,
                test_ratio=0.3,  # Sum = 1.1
                num_workers=0,
            )

    def test_create_dataloader_batch_iteration(self, sample_csv):
        """Test iterating through batches."""
        train_loader, _, _ = create_dataloader(
            data_path=sample_csv,
            target_column="price",
            batch_size=16,
            seq_len=10,
            pred_len=1,
            num_workers=0,
        )

        # Get first batch
        batch = next(iter(train_loader))
        # TimeSeriesDataset uses 'observation' and 'target'
        assert "observation" in batch
        assert "target" in batch

        # Batch should have batch dimension
        assert batch["observation"].dim() >= 2


class TestStreamingDataset:
    """Tests for StreamingDataset."""

    def test_streaming_dataset_init(self, tmp_path):
        """Test streaming dataset initialization."""
        csv_file = tmp_path / "stream.csv"
        df = pd.DataFrame({"price": range(1000)})
        df.to_csv(csv_file, index=False)

        dataset = StreamingDataset(
            csv_path=str(csv_file), target_column="price", seq_len=10, pred_len=1
        )

        assert dataset.total_length == 1000
        assert len(dataset) > 0

    def test_streaming_dataset_getitem_not_implemented(self, tmp_path):
        """Test that __getitem__ raises NotImplementedError."""
        csv_file = tmp_path / "stream.csv"
        df = pd.DataFrame({"price": range(100)})
        df.to_csv(csv_file, index=False)

        dataset = StreamingDataset(csv_path=str(csv_file), target_column="price")

        with pytest.raises(NotImplementedError):
            _ = dataset[0]
