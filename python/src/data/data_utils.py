"""
Utility functions for data loading and processing.
"""

import json
import os
from typing import Any

import pandas as pd
import torch


def read_json(json_path: str, lock: Any = None) -> Any:
    """
    Read a JSON file.

    Args:
        json_path (str): Path to the JSON file.
        lock (threading.Lock): Optional lock for thread-safe reading.

    Returns:
        dict: The loaded JSON data.
    """
    if lock is not None:
        lock.acquire(timeout=10)
    with open(json_path) as json_file:
        json_data = json.load(json_file)
    if lock is not None:
        lock.release()
    return json_data


def read_csv(csv_path: str, lock: Any = None) -> pd.DataFrame | None:
    """
    Read a CSV file into a pandas DataFrame.

    Args:
        csv_path (str): Path to the CSV file.
        lock (threading.Lock): Optional lock for thread-safe reading.

    Returns:
        pd.DataFrame: The loaded DataFrame, or None if the file doesn't exist.
    """
    if lock is not None:
        lock.acquire(timeout=10)
    df = pd.read_csv(csv_path) if os.path.isfile(csv_path) else None
    if lock is not None:
        lock.release()
    return df


def df_to_torch(
    df: pd.DataFrame, key_avoid: str, fill_nan: float | None = 0
) -> dict[str, torch.Tensor]:
    """
    Convert a pandas DataFrame to a dictionary of torch tensors.

    Args:
        df (pd.DataFrame): Input DataFrame.
        key_avoid (str): Columns containing this string will be ignored.
        fill_nan (float): Value to fill NaNs with.

    Returns:
        dict: Dictionary mapping column names to torch Tensors.
    """
    torch_dict = {}
    if fill_nan is not None:
        df = df.fillna(fill_nan)

    for col in df.columns:
        if key_avoid not in str(col):
            torch_dict[str(col)] = torch.tensor(df[col].values)
    return torch_dict
