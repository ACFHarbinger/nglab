import os
import tempfile
import unittest
from unittest.mock import MagicMock

import pandas as pd
import torch

from python.src.data.data_utils import df_to_torch, read_csv, read_json
from python.src.utils.definitions import (
    CORE_LOCK_WAIT_TIME,
    update_lock_wait_time,
)


class TestDefinitions(unittest.TestCase):
    def test_update_lock_wait_time_none(self):
        result = update_lock_wait_time(None)
        self.assertEqual(result, CORE_LOCK_WAIT_TIME)

    def test_update_lock_wait_time_with_cores(self):
        result = update_lock_wait_time(4)
        self.assertEqual(result, CORE_LOCK_WAIT_TIME * 4)


class TestDataUtils(unittest.TestCase):
    def test_read_json(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write('{"key": "value"}')
            json_path = f.name

        try:
            data = read_json(json_path)
            self.assertEqual(data, {"key": "value"})
        finally:
            if os.path.exists(json_path):
                os.unlink(json_path)

    def test_read_json_with_lock(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write('{"test": 123}')
            json_path = f.name

        mock_lock = MagicMock()
        try:
            data = read_json(json_path, lock=mock_lock)
            self.assertEqual(data, {"test": 123})
            mock_lock.acquire.assert_called_once_with(timeout=10)
            mock_lock.release.assert_called_once()
        finally:
            if os.path.exists(json_path):
                os.unlink(json_path)

    def test_read_csv(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("col1,col2\n1,2\n3,4\n")
            csv_path = f.name

        try:
            df = read_csv(csv_path)
            self.assertIsInstance(df, pd.DataFrame)
            self.assertEqual(len(df), 2)
            self.assertIn("col1", df.columns)
        finally:
            if os.path.exists(csv_path):
                os.unlink(csv_path)

    def test_read_csv_nonexistent(self):
        df = read_csv("/nonexistent/path.csv")
        self.assertIsNone(df)

    def test_read_csv_with_lock(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("a,b\n1,2\n")
            csv_path = f.name

        mock_lock = MagicMock()
        try:
            df = read_csv(csv_path, lock=mock_lock)
            self.assertIsInstance(df, pd.DataFrame)
            mock_lock.acquire.assert_called_once_with(timeout=10)
            mock_lock.release.assert_called_once()
        finally:
            if os.path.exists(csv_path):
                os.unlink(csv_path)

    def test_df_to_torch(self):
        df = pd.DataFrame(
            {
                "feature1": [1.0, 2.0, 3.0],
                "feature2": [4.0, 5.0, 6.0],
                "label_ignore": [7, 8, 9],
            }
        )

        torch_dict = df_to_torch(df, key_avoid="ignore")

        self.assertIn("feature1", torch_dict)
        self.assertIn("feature2", torch_dict)
        self.assertNotIn("label_ignore", torch_dict)
        self.assertIsInstance(torch_dict["feature1"], torch.Tensor)
        self.assertEqual(torch_dict["feature1"].shape[0], 3)

    def test_df_to_torch_with_nan(self):
        df = pd.DataFrame({"col1": [1.0, float("nan"), 3.0], "col2_skip": [4, 5, 6]})

        torch_dict = df_to_torch(df, key_avoid="skip", fill_nan=0.0)

        self.assertIn("col1", torch_dict)
        self.assertEqual(torch_dict["col1"][1].item(), 0.0)
