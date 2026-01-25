"""
Prefetching DataLoader for optimized GPU utilization.

Implements asynchronous data prefetching to overlap data loading with
GPU computation, reducing GPU idle time.
"""

import queue
import threading
from collections.abc import Iterator
from typing import Any, Union

import torch
from torch.utils.data import DataLoader, Dataset


class CUDAPrefetcher:
    """
    CUDA prefetcher that asynchronously transfers batches to GPU.

    Uses CUDA streams to overlap data transfer with computation.

    Example:
        dataloader = DataLoader(dataset, batch_size=32)
        prefetcher = CUDAPrefetcher(dataloader, device="cuda:0")

        for batch in prefetcher:
            output = model(batch)
    """

    def __init__(
        self,
        dataloader: Union[DataLoader[Any], "_IteratorWrapper"],
        device: torch.device | str = "cuda",
        prefetch_count: int = 2,
    ) -> None:
        """
        Initialize the CUDA prefetcher.

        Args:
            dataloader: Source DataLoader
            device: Target CUDA device
            prefetch_count: Number of batches to prefetch
        """
        self.dataloader = dataloader
        self.device = torch.device(device) if isinstance(device, str) else device
        self.prefetch_count = prefetch_count

        # Create CUDA stream for async transfer
        self.stream = torch.cuda.Stream(device=self.device)

        self._iterator: Iterator[Any] | None = None
        self._prefetched: list[Any] = []

    def __iter__(self) -> "CUDAPrefetcher":
        """Initialize iterator and start prefetching."""
        self._iterator = iter(self.dataloader)
        self._prefetched = []

        # Prefetch initial batches
        for _ in range(self.prefetch_count):
            self._prefetch_next()

        return self

    def __next__(self) -> Any:
        """Get next batch from prefetch queue."""
        if not self._prefetched:
            raise StopIteration

        # Wait for the current batch to finish transferring
        torch.cuda.current_stream(self.device).wait_stream(self.stream)
        batch = self._prefetched.pop(0)

        # Prefetch next batch
        self._prefetch_next()

        return batch

    def _prefetch_next(self) -> None:
        """Prefetch next batch to GPU asynchronously."""
        if self._iterator is None:
            return

        try:
            batch = next(self._iterator)
        except StopIteration:
            return

        with torch.cuda.stream(self.stream):
            batch = self._to_device(batch)

        self._prefetched.append(batch)

    def _to_device(self, data: Any) -> Any:
        """Recursively transfer data to device with non-blocking transfer."""
        if isinstance(data, torch.Tensor):
            return data.to(self.device, non_blocking=True)
        elif isinstance(data, dict):
            return {k: self._to_device(v) for k, v in data.items()}
        elif isinstance(data, list | tuple):
            return type(data)(self._to_device(v) for v in data)
        return data

    def __len__(self) -> int:
        """Return length of underlying dataloader."""
        return len(self.dataloader)


class BackgroundPrefetcher:
    """
    Background thread prefetcher for CPU-bound data loading.

    Uses a background thread to load data while the main thread processes.
    Useful when data loading/preprocessing is the bottleneck.

    Example:
        dataloader = DataLoader(dataset, batch_size=32)
        prefetcher = BackgroundPrefetcher(dataloader, prefetch_count=4)

        for batch in prefetcher:
            output = model(batch.to("cuda"))
    """

    def __init__(
        self,
        dataloader: DataLoader[Any],
        prefetch_count: int = 4,
        timeout: float = 60.0,
    ) -> None:
        """
        Initialize the background prefetcher.

        Args:
            dataloader: Source DataLoader
            prefetch_count: Number of batches to prefetch
            timeout: Timeout for getting items from queue
        """
        self.dataloader = dataloader
        self.prefetch_count = prefetch_count
        self.timeout = timeout

        self._queue: queue.Queue[Any] = queue.Queue(maxsize=prefetch_count)
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._exception: Exception | None = None

    def __iter__(self) -> "BackgroundPrefetcher":
        """Start background thread and return iterator."""
        self._stop_event.clear()
        self._exception = None
        self._queue = queue.Queue(maxsize=self.prefetch_count)

        self._thread = threading.Thread(target=self._prefetch_worker, daemon=True)
        self._thread.start()

        return self

    def __next__(self) -> Any:
        """Get next batch from prefetch queue."""
        # Check for exceptions from background thread
        if self._exception is not None:
            raise self._exception

        try:
            item = self._queue.get(timeout=self.timeout)
        except queue.Empty:
            raise RuntimeError("Prefetcher timeout waiting for batch") from None

        if item is StopIteration:
            # Cleanup
            self._stop_event.set()
            if self._thread is not None:
                self._thread.join(timeout=1.0)

            # Check for exceptions again
            if self._exception is not None:
                raise self._exception

            raise StopIteration

        return item

    def _prefetch_worker(self) -> None:
        """Background worker that loads batches into the queue."""
        try:
            for batch in self.dataloader:
                if self._stop_event.is_set():
                    break
                self._queue.put(batch)

            # Signal end of data
            self._queue.put(StopIteration)
        except Exception as e:
            self._exception = e
            self._queue.put(StopIteration)

    def __len__(self) -> int:
        """Return length of underlying dataloader."""
        return len(self.dataloader)

    def __del__(self) -> None:
        """Cleanup background thread."""
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)


class PrefetchDataLoader(DataLoader[Any]):
    """
    DataLoader with built-in CUDA prefetching.

    Extends PyTorch DataLoader with automatic GPU prefetching
    and pinned memory for optimal performance.

    Example:
        dataloader = PrefetchDataLoader(
            dataset,
            batch_size=32,
            num_workers=4,
            device="cuda:0",
        )

        for batch in dataloader:
            # batch is already on GPU
            output = model(batch)
    """

    def __init__(  # noqa: PLR0913
        self,
        dataset: Dataset[Any],
        batch_size: int = 1,
        shuffle: bool = False,
        num_workers: int = 0,
        pin_memory: bool = True,
        device: str | None = None,
        prefetch_factor: int = 2,
        persistent_workers: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the prefetching DataLoader.

        Args:
            dataset: Source dataset
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes
            pin_memory: Use pinned memory for faster GPU transfer
            device: Target device (None for CPU, "cuda" for GPU)
            prefetch_factor: Number of batches to prefetch per worker
            persistent_workers: Keep workers alive between epochs
            **kwargs: Additional DataLoader arguments
        """
        # Enable pinned memory if using GPU
        if device is not None and "cuda" in device:
            pin_memory = True

        # Set prefetch_factor only if num_workers > 0
        if num_workers > 0:
            kwargs["prefetch_factor"] = prefetch_factor
            kwargs["persistent_workers"] = persistent_workers

        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            **kwargs,
        )

        self.device = torch.device(device) if device else None
        self._prefetcher: CUDAPrefetcher | None = None

    def __iter__(self) -> Iterator[Any]:  # type: ignore[override]
        """Return iterator with optional CUDA prefetching."""
        if self.device is not None and self.device.type == "cuda":
            # Use CUDA prefetcher
            base_iter = super().__iter__()
            # Create a wrapper DataLoader for the prefetcher
            return CUDAPrefetcher(
                _IteratorWrapper(base_iter, len(self)),
                device=self.device,
            )
        else:
            return super().__iter__()


class _IteratorWrapper:
    """Wrapper to make an iterator look like a DataLoader for CUDAPrefetcher."""

    def __init__(self, iterator: Iterator[Any], length: int) -> None:
        """Initialize IteratorWrapper."""
        self._iterator = iterator
        self._length = length

    def __iter__(self) -> Iterator[Any]:
        """Return the iterator."""
        return self._iterator

    def __len__(self) -> int:
        """Return the total number of batches."""
        return self._length


def create_optimized_dataloader(  # noqa: PLR0913
    dataset: Dataset[Any],
    batch_size: int = 32,
    num_workers: int | None = None,
    device: str = "cuda",
    shuffle: bool = True,
    drop_last: bool = True,
    **kwargs: Any,
) -> DataLoader[Any]:
    """
    Create an optimized DataLoader with best practices for GPU training.

    Automatically configures:
    - Number of workers based on CPU count
    - Pinned memory for GPU training
    - Persistent workers for efficiency
    - Prefetch factor for overlapping data loading

    Args:
        dataset: Source dataset
        batch_size: Batch size
        num_workers: Number of workers (auto-detected if None)
        device: Target device
        shuffle: Whether to shuffle data
        drop_last: Drop incomplete last batch
        **kwargs: Additional DataLoader arguments

    Returns:
        Optimized DataLoader instance
    """
    import os

    # Auto-detect number of workers
    if num_workers is None:
        # Use 4 workers per GPU, capped at CPU count
        cpu_count = os.cpu_count() or 4
        num_workers = min(4, cpu_count)

    # Configure based on device
    use_cuda = "cuda" in device and torch.cuda.is_available()

    return PrefetchDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=use_cuda,
        device=device if use_cuda else None,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
        prefetch_factor=2,
        **kwargs,
    )


def benchmark_dataloader(
    dataloader: DataLoader[Any],
    num_batches: int = 100,
    warmup_batches: int = 10,
) -> dict[str, Any]:
    """
    Benchmark DataLoader throughput.

    Args:
        dataloader: DataLoader to benchmark
        num_batches: Number of batches to time
        warmup_batches: Number of warmup batches

    Returns:
        Dictionary with timing statistics
    """
    import time

    # Warmup
    data_iter = iter(dataloader)
    for _ in range(warmup_batches):
        try:
            _ = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            _ = next(data_iter)

    # Benchmark
    times = []
    batch_sizes = []

    for _ in range(num_batches):
        start = time.perf_counter()
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)  # Convert to ms

        # Get batch size
        if isinstance(batch, torch.Tensor):
            batch_sizes.append(batch.shape[0])
        elif isinstance(batch, list | tuple):
            batch_sizes.append(batch[0].shape[0])
        elif isinstance(batch, dict):
            first_key = next(iter(batch.keys()))
            batch_sizes.append(batch[first_key].shape[0])

    times_tensor = torch.tensor(times)
    avg_batch_size = sum(batch_sizes) / len(batch_sizes)

    return {
        "mean_latency_ms": times_tensor.mean().item(),
        "std_latency_ms": times_tensor.std().item(),
        "min_latency_ms": times_tensor.min().item(),
        "max_latency_ms": times_tensor.max().item(),
        "throughput_batches_per_sec": 1000 / times_tensor.mean().item(),
        "throughput_samples_per_sec": avg_batch_size
        * 1000
        / times_tensor.mean().item(),
        "avg_batch_size": avg_batch_size,
        "num_batches": num_batches,
    }
