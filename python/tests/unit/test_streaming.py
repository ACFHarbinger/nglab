from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from python.src.data.streaming import StreamingFinancialDataset


@pytest.fixture
def csv_file(tmp_path):
    df = pd.DataFrame(
        {
            "open": np.random.randn(100),
            "close": np.random.randn(100),
            "volume": np.random.randn(100),
        }
    )
    file_path = tmp_path / "test.csv"
    df.to_csv(file_path, index=False)
    return file_path


@pytest.fixture
def parquet_file(tmp_path):
    pd.DataFrame({"open": np.random.randn(100), "close": np.random.randn(100)})
    file_path = tmp_path / "test.parquet"
    # We might not have pyarrow installed in the test env, so we'll just create the path
    # and mock the reading logic in the test.
    file_path.touch()
    return file_path


def test_streaming_financial_dataset_init_not_found():
    with pytest.raises(FileNotFoundError):
        StreamingFinancialDataset("non_existent.csv")


def test_streaming_financial_dataset_csv(csv_file):
    dataset = StreamingFinancialDataset(csv_file, chunk_size=20)

    items = list(dataset)
    assert len(items) == 100
    assert isinstance(items[0], dict)
    assert "open" in items[0]
    assert "close" in items[0]


def test_streaming_financial_dataset_transform(csv_file):
    def transform(item):
        item["new_col"] = item["open"] * 2
        return item

    dataset = StreamingFinancialDataset(csv_file, chunk_size=20, transform=transform)
    items = list(dataset)
    assert "new_col" in items[0]
    assert items[0]["new_col"] == items[0]["open"] * 2


def test_streaming_financial_dataset_shuffle(csv_file):
    # Use a small chunk size and large shuffle buffer
    dataset = StreamingFinancialDataset(csv_file, chunk_size=10, shuffle_buffer_size=50)

    items = list(dataset)
    assert len(items) == 100

    # Check if they are shuffled (at least not in the same order as original)
    df = pd.read_csv(csv_file)
    original_opens = df["open"].tolist()
    items_opens = [item["open"] for item in items]

    assert items_opens != original_opens


def test_streaming_financial_dataset_loop_forever(csv_file):
    dataset = StreamingFinancialDataset(csv_file, chunk_size=50, loop_forever=True)

    count = 0
    for _ in dataset:
        count += 1
        if count >= 150:
            break
    assert count == 150


def test_streaming_financial_dataset_multi_worker_warning(csv_file):
    dataset = StreamingFinancialDataset(csv_file)
    with patch("python.src.data.streaming.get_worker_info") as mock_worker_info:
        mock_info = MagicMock()
        mock_info.num_workers = 2
        mock_worker_info.return_value = mock_info

        with patch("python.src.data.streaming.logger.warning") as mock_warning:
            it = iter(dataset)
            next(it)
            mock_warning.assert_called_once()
            assert "duplicated" in mock_warning.call_args[0][0]


@patch("pyarrow.parquet.ParquetFile")
def test_streaming_financial_dataset_parquet(mock_parquet_file, parquet_file):
    # Mocking pyarrow behavior
    mock_file_instance = MagicMock()
    mock_parquet_file.return_value = mock_file_instance

    mock_batch = MagicMock()
    mock_batch.to_pandas.return_value = pd.DataFrame({"open": [1.0, 2.0]})
    mock_file_instance.iter_batches.return_value = [mock_batch]

    dataset = StreamingFinancialDataset(
        parquet_file, chunk_size=2, file_format="parquet"
    )
    items = list(dataset)

    assert len(items) == 2
    assert items[0]["open"] == 1.0
    mock_file_instance.iter_batches.assert_called_once_with(batch_size=2)


def test_unsupported_format(tmp_path):
    f = tmp_path / "test.txt"
    f.touch()
    dataset = StreamingFinancialDataset(f, file_format="txt")
    with pytest.raises(ValueError, match="Unsupported file format"):
        next(iter(dataset))
