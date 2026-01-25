"""
Polymarket Dataset Module.
Handles loading and processing of multivariate time-series data from Polymarket events.
"""

import os
from collections.abc import Callable

import pandas as pd
import torch

from .data_utils import read_json


class PolymarketDataset(torch.utils.data.Dataset[dict[str, torch.Tensor]]):
    """
    Dataset class for Polymarket prediction data (Multivariate).
    Loads multiple candidate files and aligns them by timestamp.
    """

    def __init__(  # noqa: PLR0913
        self,
        name: str,
        dataset_dir: str,
        seq_len: int,
        pred_len: int,
        download: bool = False,
        transform: (
            Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]] | None
        ) = None,
    ) -> None:
        """
        Initialize the dataset.

        Args:
            name (str): Dataset name.
            dataset_dir (str): Directory containing the data.
            seq_len (int): Input sequence length.
            pred_len (int): Prediction sequence length.
            download (bool): Whether to download data (not implemented).
            transform (callable): Optional transform to apply.
        """
        super().__init__()
        if download:
            self._download()

        self.name = name
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.transform = transform
        self.dataset_dir = dataset_dir

        self.dataset: dict[str, torch.Tensor] = {}
        self.dataset_len: int = 0
        self._load_multivariate_data()

    def _download(self) -> None:
        raise NotImplementedError

    def __len__(self) -> int:
        """Return the total number of samples."""
        return self.dataset_len

    def _get_name(self) -> str:
        return self.name

    def _load_multivariate_data(self) -> None:
        """
        Loads all CSVs defined in metadata.json and merges them into a single multivariate tensor.
        """
        metadata_path = os.path.join(self.dataset_dir, "metadata.json")
        metadata = read_json(metadata_path)

        # Sort metadata by id to ensure consistent column ordering
        # (assuming 5 candidates, e.g. ids 0-4)
        metadata = sorted(metadata, key=lambda x: x["id"])

        data_frames = []
        for md in metadata:
            filepath = os.path.join(self.dataset_dir, md["filename"])
            # Read CSV using pandas directly for easier merging
            df = pd.read_csv(filepath)

            # Ensure Timestamp is datetime
            if "Timestamp (UTC)" in df.columns:
                df["Timestamp (UTC)"] = pd.to_datetime(df["Timestamp (UTC)"])
                df = df.set_index("Timestamp (UTC)")
                df = df.sort_index()

            # Keep only the Price column and rename it to the candidate ID/Name
            # Assuming 'Price' column exists
            if "Price" in df.columns:
                df = df[["Price"]]
                df.columns = [f"Candidate_{md['id']}"]
                data_frames.append(df)

        if not data_frames:
            raise ValueError("No valid data frames loaded from metadata.")

        # Merge all dataframes on the index (Timestamp)
        # using outer join to capture all time steps, then forward fill
        merged_df = pd.concat(data_frames, axis=1)  # concat on axis 1 aligns on index
        merged_df = merged_df.sort_index()
        merged_df = merged_df.fillna(method="ffill").fillna(
            method="bfill"
        )  # Handle initial NaNs

        # Convert to tensor [TotalSteps, NumCandidates]
        price_tensor = torch.tensor(merged_df.values, dtype=torch.float32)

        # Create sequences
        # We need to slice this tensor into overlapping windows
        # Input: [SeqLen, Features]
        # Target: [PredLen, Features] (Many-to-Many prediction)

        num_samples = len(price_tensor) - self.seq_len - self.pred_len + 1
        if num_samples <= 0:
            raise ValueError(
                f"Dataset too short ({len(price_tensor)}) for seq_len={self.seq_len} + pred_len={self.pred_len}"
            )

        # Efficiently create windows using unfold or list comprehension
        # (B, L, D)

        # Inputs (X):
        # We want X[i] = price_tensor[i : i+seq_len]

        # Targets (Y):
        # We want Y[i] = price_tensor[i+seq_len : i+seq_len+pred_len]

        X_list = []
        Y_list = []

        for i in range(num_samples):
            x = price_tensor[i : i + self.seq_len]
            y = price_tensor[i + self.seq_len : i + self.seq_len + self.pred_len]
            X_list.append(x)
            Y_list.append(y)

        self.dataset["Price"] = torch.stack(X_list)
        self.dataset["Labels"] = torch.stack(Y_list)

        self.dataset_len = len(self.dataset["Price"])
        print(f"Loaded Multivariate Dataset: Shape {self.dataset['Price'].shape}")

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """
        Get a sample by index.
        """
        data = {}
        # Clone to avoid potential side effects if transforms modify in place (though typically they don't)
        data["Price"] = self.dataset["Price"][index].clone()
        data["Labels"] = self.dataset["Labels"][index].clone()

        if self.transform is not None:
            data = self.transform(data)
        return data
