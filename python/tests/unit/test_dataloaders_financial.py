import os
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from python.src.data.dataloaders import (
    FinancialDataset,
    StreamingDataset,
    create_dataloader,
)


class TestFinancialDataset(unittest.TestCase):
    def setUp(self):
        # Create a temporary CSV file for testing
        self.temp_csv = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        )
        self.temp_csv.write("date,price,volume\n")
        for i in range(100):
            self.temp_csv.write(f"2024-01-{i + 1:02d},{100 + i},{1000 + i * 10}\n")
        self.temp_csv.close()

    def tearDown(self):
        os.unlink(self.temp_csv.name)

    def test_financial_dataset_init(self):
        dataset = FinancialDataset(
            csv_path=self.temp_csv.name,
            target_column="price",
            seq_len=10,
            pred_len=1,
            add_technical_indicators=False,
            multi_asset=False,
        )

        self.assertFalse(dataset.add_technical_indicators)
        self.assertFalse(dataset.multi_asset)
        self.assertIsNotNone(dataset.data)

    def test_financial_dataset_with_normalization(self):
        dataset = FinancialDataset(
            csv_path=self.temp_csv.name,
            target_column="price",
            seq_len=10,
            pred_len=1,
            normalize="minmax",
        )

        self.assertIsNotNone(dataset.raw_min)
        self.assertIsNotNone(dataset.raw_max)


class TestCreateDataloader(unittest.TestCase):
    def setUp(self):
        self.temp_csv = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        )
        self.temp_csv.write("date,price,volume\n")
        for i in range(200):
            self.temp_csv.write(f"2024-01-{i + 1:02d},{100 + i},{1000 + i * 10}\n")
        self.temp_csv.close()

    def tearDown(self):
        os.unlink(self.temp_csv.name)

    def test_create_dataloader_csv(self):
        train_loader, val_loader, test_loader = create_dataloader(
            data_path=self.temp_csv.name,
            target_column="price",
            batch_size=16,
            seq_len=10,
            pred_len=1,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            format="csv",
            num_workers=0,  # Set to 0 for testing
        )

        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(val_loader)
        self.assertIsNotNone(test_loader)
        self.assertEqual(train_loader.batch_size, 16)

    def test_create_dataloader_ratios_validation(self):
        with self.assertRaises(AssertionError):
            create_dataloader(
                data_path=self.temp_csv.name,
                target_column="price",
                train_ratio=0.5,
                val_ratio=0.3,
                test_ratio=0.3,  # Sum > 1.0
                format="csv",
            )

    @patch("pandas.read_parquet")
    def test_create_dataloader_parquet(self, mock_read_parquet):
        df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=100),
                "price": range(100, 200),
                "volume": range(1000, 1100),
            }
        )
        mock_read_parquet.return_value = df

        with tempfile.NamedTemporaryFile(
            suffix=".parquet", delete=False
        ) as temp_parquet:
            temp_parquet_path = temp_parquet.name

        try:
            train_loader, _val_loader, _test_loader = create_dataloader(
                data_path=temp_parquet_path,
                target_column="price",
                batch_size=16,
                format="parquet",
                num_workers=0,
            )

            self.assertIsNotNone(train_loader)
        finally:
            # Cleanup created CSV
            csv_path = temp_parquet_path.replace(".parquet", ".csv")
            if os.path.exists(csv_path):
                os.unlink(csv_path)
            if os.path.exists(temp_parquet_path):
                os.unlink(temp_parquet_path)

    def test_create_dataloader_unsupported_format(self):
        with self.assertRaises(ValueError):
            create_dataloader(
                data_path=self.temp_csv.name,
                target_column="price",
                format="unsupported",
            )


class TestStreamingDataset(unittest.TestCase):
    def setUp(self):
        self.temp_csv = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        )
        self.temp_csv.write("date,price,volume\n")
        for i in range(100):
            self.temp_csv.write(f"2024-01-{i + 1:02d},{100 + i},{1000 + i * 10}\n")
        self.temp_csv.close()

    def tearDown(self):
        os.unlink(self.temp_csv.name)

    def test_streaming_dataset_length(self):
        dataset = StreamingDataset(
            csv_path=self.temp_csv.name, target_column="price", seq_len=10, pred_len=1
        )

        expected_length = max(
            0, 100 - 10 - 1 + 1
        )  # total_length - seq_len - pred_len + 1
        self.assertEqual(len(dataset), expected_length)

    def test_streaming_dataset_getitem_not_implemented(self):
        dataset = StreamingDataset(
            csv_path=self.temp_csv.name, target_column="price", seq_len=10, pred_len=1
        )

        with self.assertRaises(NotImplementedError):
            _ = dataset[0]
