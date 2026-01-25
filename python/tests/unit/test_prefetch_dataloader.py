from unittest.mock import MagicMock, patch

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from python.src.data.prefetch_dataloader import (
    BackgroundPrefetcher,
    CUDAPrefetcher,
    PrefetchDataLoader,
    _IteratorWrapper,
    benchmark_dataloader,
    create_optimized_dataloader,
)


class SimpleDataset(Dataset):
    def __init__(self, size=10):
        self.size = size
        self.data = [torch.randn(3, 3) for _ in range(size)]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]


@pytest.fixture
def dataset():
    return SimpleDataset()


@pytest.fixture
def dataloader(dataset):
    return DataLoader(dataset, batch_size=2)


def test_iterator_wrapper():
    items = [1, 2, 3]
    it = iter(items)
    wrapper = _IteratorWrapper(it, len(items))
    assert len(wrapper) == 3
    assert list(wrapper) == [1, 2, 3]


def test_cudaprefetcher_init(dataloader):
    with patch("torch.cuda.Stream") as mock_stream:
        prefetcher = CUDAPrefetcher(dataloader, device="cuda:0", prefetch_count=2)
        assert prefetcher.device == torch.device("cuda:0")
        assert prefetcher.prefetch_count == 2
        mock_stream.assert_called_once_with(device=torch.device("cuda:0"))


def test_cudaprefetcher_to_device():
    with patch("torch.cuda.Stream"):
        prefetcher = CUDAPrefetcher(
            MagicMock(), device="cpu"
        )  # Use cpu for simple test

        # Test tensor
        t = torch.randn(2)
        moved_t = prefetcher._to_device(t)
        assert torch.equal(t, moved_t)

        # Test dict
        d = {"a": torch.randn(2), "b": [torch.randn(2)]}
        moved_d = prefetcher._to_device(d)
        assert isinstance(moved_d, dict)
        assert torch.equal(d["a"], moved_d["a"])
        assert torch.equal(d["b"][0], moved_d["b"][0])

        # Test list/tuple
        test_list = [torch.randn(2), (torch.randn(2),)]
        moved_list = prefetcher._to_device(test_list)
        assert isinstance(moved_list, list)
        assert isinstance(moved_list[1], tuple)


@patch("torch.cuda.current_stream")
@patch("torch.cuda.stream")
@patch("torch.cuda.Stream")
def test_cudaprefetcher_iter_next(
    mock_stream_class, mock_stream_context, mock_current_stream, dataloader
):
    # Mock stream behavior
    mock_stream = MagicMock()
    mock_stream_class.return_value = mock_stream

    prefetcher = CUDAPrefetcher(dataloader, device="cuda:0", prefetch_count=2)

    # Mock _to_device to just return data
    with patch.object(CUDAPrefetcher, "_to_device", side_effect=lambda x: x):
        items = []
        for batch in prefetcher:
            items.append(batch)

        assert len(items) == 5  # 10 items, batch_size 2
        assert mock_stream_context.call_count == 5
        assert mock_current_stream.return_value.wait_stream.call_count == 5


def test_background_prefetcher_init(dataloader):
    prefetcher = BackgroundPrefetcher(dataloader, prefetch_count=2)
    assert prefetcher.prefetch_count == 2
    assert len(prefetcher) == 5


def test_background_prefetcher_iter_next(dataloader):
    prefetcher = BackgroundPrefetcher(dataloader, prefetch_count=2)
    items = []
    for batch in prefetcher:
        items.append(batch)
    assert len(items) == 5
    assert not prefetcher._thread.is_alive()


def test_background_prefetcher_timeout(dataloader):
    prefetcher = BackgroundPrefetcher(dataloader, timeout=0.01)
    with patch.object(prefetcher, "_prefetch_worker"):
        # Don't start the worker properly to trigger timeout
        prefetcher._thread = MagicMock()
        prefetcher._thread.is_alive.return_value = True
        with pytest.raises(RuntimeError, match="Prefetcher timeout waiting for batch"):
            next(iter(prefetcher))


def test_background_prefetcher_exception(dataset):
    class FailingDataset(Dataset):
        def __len__(self):
            return 1

        def __getitem__(self, idx):
            raise ValueError("Test Error")

    dl = DataLoader(FailingDataset())
    prefetcher = BackgroundPrefetcher(dl)
    with pytest.raises(ValueError, match="Test Error"):
        for _ in prefetcher:
            pass


def test_prefetch_dataloader_init(dataset):
    # CPU case
    dl = PrefetchDataLoader(dataset, batch_size=2, device=None)
    assert dl.device is None

    # CUDA case
    with patch("torch.cuda.is_available", return_value=True):
        dl_cuda = PrefetchDataLoader(
            dataset, batch_size=2, device="cuda:0", num_workers=2
        )
        assert dl_cuda.device == torch.device("cuda:0")
        assert dl_cuda.pin_memory is True


def test_prefetch_dataloader_iter(dataset):
    # CPU case
    dl = PrefetchDataLoader(dataset, batch_size=2)
    it = iter(dl)
    assert not isinstance(it, CUDAPrefetcher)

    # CUDA case
    with patch("torch.cuda.is_available", return_value=True):
        with patch(
            "python.src.data.prefetch_dataloader.CUDAPrefetcher"
        ) as mock_prefetcher:
            dl_cuda = PrefetchDataLoader(dataset, batch_size=2, device="cuda:0")
            iter(dl_cuda)
            mock_prefetcher.assert_called_once()


def test_create_optimized_dataloader(dataset):
    with patch("os.cpu_count", return_value=8):
        # Auto workers
        dl = create_optimized_dataloader(dataset, device="cpu")
        assert dl.num_workers == 4

        # CUDA optimized
        with patch("torch.cuda.is_available", return_value=True):
            dl_cuda = create_optimized_dataloader(dataset, device="cuda:0")
            assert dl_cuda.pin_memory is True
            assert dl_cuda.device == torch.device("cuda:0")


def test_benchmark_dataloader(dataloader):
    stats = benchmark_dataloader(dataloader, num_batches=2, warmup_batches=1)
    assert "mean_latency_ms" in stats
    assert "throughput_batches_per_sec" in stats
    assert stats["num_batches"] == 2
    assert stats["avg_batch_size"] == 2


def test_benchmark_dataloader_complex_batch(dataset):
    class ComplexDataset(Dataset):
        def __len__(self):
            return 10

        def __getitem__(self, idx):
            return {"data": torch.randn(2, 3), "label": torch.tensor([1])}

    dl = DataLoader(ComplexDataset(), batch_size=2)
    stats = benchmark_dataloader(dl, num_batches=2, warmup_batches=1)
    assert stats["avg_batch_size"] == 2
