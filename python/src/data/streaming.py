"""
Streaming Data Loading for Large Financial Datasets.

Provides IterableDatasets for handling datasets that are too large to fit in memory
by streaming data in chunks from disk.
"""

import logging
import random
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from torch.utils.data import IterableDataset, get_worker_info

logger = logging.getLogger(__name__)


class StreamingFinancialDataset(IterableDataset[dict[str, Any]]):
    """
    Iterable Dataset for streaming large financial datasets from CSV/Parquet files.

    Supports:
    - Chunked reading to minimize memory usage
    - Multi-worker support (splitting files or chunks across workers)
    - Infinite looping options
    - Preprocessing hooks
    """

    def __init__(  # noqa: PLR0913
        self,
        filepath: str | Path,
        chunk_size: int = 10000,
        loop_forever: bool = False,
        shuffle_buffer_size: int = 0,
        transform: Any | None = None,
        file_format: str = "csv",
        **reader_kwargs: Any,
    ) -> None:
        """
        Initialize StreamingFinancialDataset.

        Args:
            filepath: Path to the data file.
            chunk_size: Number of rows to read per chunk.
            loop_forever: If True, iterate indefinitely.
            shuffle_buffer_size: If > 0, maintain a buffer to shuffle data locally.
            transform: Optional transform to apply to each sample.
            file_format: "csv" or "parquet".
            reader_kwargs: Additional arguments passed to pd.read_csv/read_parquet.
        """
        self.filepath = Path(filepath)
        self.chunk_size = chunk_size
        self.loop_forever = loop_forever
        self.shuffle_buffer_size = shuffle_buffer_size
        self.transform = transform
        self.file_format = file_format.lower()
        self.reader_kwargs = reader_kwargs

        if not self.filepath.exists():
            raise FileNotFoundError(f"Data file not found: {self.filepath}")

    def _get_iterator(self) -> Iterator[pd.DataFrame]:
        """Get an iterator over chunks of the dataframe."""
        if self.file_format == "csv":
            reader = pd.read_csv(
                self.filepath, chunksize=self.chunk_size, **self.reader_kwargs
            )
            return cast(Iterator[pd.DataFrame], reader)
        elif self.file_format == "parquet":
            # Parquet file reading is typically not chunked by lines in the same way as CSV
            # unless using pyarrow.parquet.ParquetFile.iter_batches().
            # For simplicity with pandas, we might need a workaround or assume PyArrow engine.
            # Here we implement a basic PyArrow batch reader if installed, else fail.
            try:
                import pyarrow.parquet as pq

                parquet_file = pq.ParquetFile(self.filepath)
                # Convert pyarrow batches to pandas
                return (
                    batch.to_pandas()
                    for batch in parquet_file.iter_batches(batch_size=self.chunk_size)
                )
            except ImportError:
                logger.error("PyArrow is required for streaming Parquet files.")
                raise
        else:
            raise ValueError(f"Unsupported file format: {self.file_format}")

    def _process_chunk(self, chunk: pd.DataFrame) -> Iterator[dict[str, Any]]:
        """Yield individual samples from a chunk."""
        # Convert to dict records or keep as tensor rows?
        # Usually datasets yield individual items.

        # If we have a huge shuffle buffer, we would add to buffer here.
        # For simplicity, we iterate rows.
        for _, row in chunk.iterrows():
            item = cast(
                dict[str, Any], row.to_dict()
            )  # Or convert to Tensor directly if faster
            # Simple numeric conversion could happen here
            if self.transform:
                item = self.transform(item)
            yield item

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over the dataset."""
        worker_info = get_worker_info()

        # NOTE: For a single huge file, splitting work among workers is tricky without
        # distinct files. Common strategy:
        # 1. Each worker reads the whole file but skips items (inefficient IO).
        # 2. Each worker reads a distinct subset of files (if list of files).
        # 3. Only Worker 0 reads and queues? (Not standard DataLoader)
        #
        # Here we assume a single file. Handling offsets cleanly in CSV is hard.
        # A simple naive approach for CSV:
        # Worker i processes every Nth chunk? Or we just duplicate data if not careful.
        #
        # Improved Strategy:
        # If we had multiple files, we'd split file list.
        # With one file, we might warn or just accept that multiple workers duplicate data
        # unless manual sharding logic is added (e.g. byte offsets).
        #
        # For now, we will log a warning if standard workers > 0 and single file.

        if worker_info is not None and worker_info.num_workers > 1:
            logger.warning(
                f"StreamingFinancialDataset used with {worker_info.num_workers} workers on a single file. "
                "Data might be duplicated or handling split inefficiently."
            )

        # Shuffle buffer state
        buffer: list[dict[str, Any]] = []

        def iterator_logic() -> Iterator[dict[str, Any]]:
            """Core logic for iterating and optional shuffling."""
            dataset_iter = self._get_iterator()

            for chunk in dataset_iter:
                # Naive shuffling: Accumulate a buffer
                if self.shuffle_buffer_size > 0:
                    for _, row in chunk.iterrows():
                        item = cast(dict[str, Any], row.to_dict())
                        if self.transform:
                            item = self.transform(item)

                        if len(buffer) < self.shuffle_buffer_size:
                            buffer.append(item)
                        else:
                            # Swap with random element and yield
                            idx = np.random.randint(0, len(buffer))
                            yield buffer[idx]
                            buffer[idx] = item
                else:
                    # Direct yield
                    yield from self._process_chunk(chunk)

            # Yield remaining buffer
            if self.shuffle_buffer_size > 0:
                random.shuffle(buffer)
                yield from buffer
                buffer.clear()

        if self.loop_forever:
            while True:
                yield from iterator_logic()
        else:
            yield from iterator_logic()
