"""
GPU-accelerated Feature Engineering Utilities.

Provides PyTorch-based implementations of common technical indicators
for calculating features directly on the GPU, avoiding CPU-GPU transfers.
"""

from typing import Any

import torch


class GPUFeatureEngineer:
    """
    Feature engineering optimized for GPU execution using PyTorch operations.
    Can be used with CPU tensors as well, but optimized for CUDA.
    """

    def __init__(self, device: str | torch.device | None = None) -> None:
        """
        Initialize the engineer.

        Args:
            device: Target device (e.g. "cuda", "cpu"). If None, uses input tensor device.
        """
        self.device = device

    def _to_tensor(self, data: torch.Tensor | list[Any]) -> torch.Tensor:
        """Convert input to tensor on correct device."""
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)

        if self.device is not None:
            if data.device != self.device:
                data = data.to(self.device)
        return data.float()

    def moving_average(self, data: torch.Tensor, window: int) -> torch.Tensor:
        """
        Calculate Simple Moving Average (SMA).

        Args:
            data: Input tensor (1D or 2D [batch, time])
            window: Window size

        Returns:
            Tensor of same shape, padded with NaNs at start
        """
        data = self._to_tensor(data)

        # Kernel for convolution
        kernel = torch.ones(window, device=data.device) / window
        kernel = kernel.view(1, 1, window)

        # Reshape for conv1d: [batch, channels, length]
        if data.dim() == 1:
            x = data.view(1, 1, -1)
        else:
            x = data.view(data.shape[0], 1, data.shape[1])

        # Padding for causal convolution (only look back)
        # Pad left by window-1
        x_pad = torch.nn.functional.pad(x, (window - 1, 0))

        # Conv1d
        ma = torch.nn.functional.conv1d(x_pad, kernel)

        # Reshape back
        if data.dim() == 1:
            return ma.view(-1)
        else:
            return ma.view(data.shape[0], -1)

    def exponential_moving_average(self, data: torch.Tensor, span: int) -> torch.Tensor:
        """
        Calculate Exponential Moving Average (EMA).
        Note: This is an iterative implementation for correctness,
        which might be slower than convolution approximation but is exact.
        For very long sequences, a scan/cumsum approach could be faster.
        """
        data = self._to_tensor(data)
        alpha = 2 / (span + 1)

        # PyTorch iterative approach using weighted cumsum logic is complex to vectorize efficiently
        # without specialized kernels. For moderate sequence lengths, a loop or cumprod approach works.
        # Here we use a widely used vectorized approximation or simple loop if JITable.

        # Let's use a simpler decay approach which is vectorized:
        # EMA_t = alpha * x_t + (1-alpha) * EMA_{t-1}
        # This can be computed as a convolution with decaying powers if infinite history,
        # but for finite finite window it's tricky.

        # A simple Python loop on tensors is slow.
        # However, for GPU, we want to avoid Python loops.
        # We can implement this using log-space cumsum for numerical stability and potential vectorization.

        # y_t = sum_{i=0}^t x_i * alpha * (1-alpha)^{t-i} + (1-alpha)^{t+1} * y_{-1}

        # Let's use a simplified approach:
        # We'll stick to a loop if compiled with torch.jit or assume short sequences?
        # Actually, for "GPU Feature Engineering", performance matters.
        # Let's use a recurrence relation that can be unrolled or a simple loop for now as a baseline,
        # acknowledging typical financial time series are not massive (thousands points).

        if data.dim() == 1:
            # Add batch dim
            data = data.unsqueeze(0)

        batch_size, seq_len = data.shape
        ema = torch.zeros_like(data)

        # Initial value
        ema[:, 0] = data[:, 0]

        # Sequential update: inevitable without specialized scan kernel
        # But we can process full batch in parallel
        # (1-alpha)
        decay = 1.0 - alpha

        # Simple loop
        curr = data[:, 0]
        for t in range(1, seq_len):
            curr = alpha * data[:, t] + decay * curr
            ema[:, t] = curr

        if data.shape[0] == 1 and batch_size == 1:
            return ema.squeeze(0)

        return ema.view(data.shape)

    def rsi(self, data: torch.Tensor, window: int = 14) -> torch.Tensor:
        """
        Calculate Relative Strength Index (RSI).
        """
        data = self._to_tensor(data)

        delta = data.diff(dim=-1)
        # Pad first element to keep length
        if data.dim() == 1:
            delta = torch.cat([torch.zeros(1, device=data.device), delta])
        else:
            delta = torch.cat(
                [torch.zeros((data.shape[0], 1), device=data.device), delta], dim=1
            )

        gain = torch.max(delta, torch.zeros_like(delta))
        loss = torch.abs(torch.min(delta, torch.zeros_like(delta)))

        # Use EMA for smoothing (Wilder's Smoothing) usually 1/window
        # But standard RSI often uses SMA first then EMA.
        # Common def: sum gains / window.

        # Simple implementation: SMA of gains / SMA of losses
        avg_gain = self.moving_average(gain, window)
        avg_loss = self.moving_average(loss, window)

        rs = avg_gain / (avg_loss + 1e-10)
        rsi_val = 100 - (100 / (1 + rs))

        return rsi_val

    def macd(
        self, data: torch.Tensor, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate MACD, Signal line, and Histogram.
        """
        ema_fast = self.exponential_moving_average(data, fast)
        ema_slow = self.exponential_moving_average(data, slow)

        macd_line = ema_fast - ema_slow
        signal_line = self.exponential_moving_average(macd_line, signal)
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    def bollinger_bands(
        self, data: torch.Tensor, window: int = 20, num_std: float = 2.0
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate Bollinger Bands (Upper, Middle, Lower).
        """
        data = self._to_tensor(data)

        sma = self.moving_average(data, window)

        # Standard Deviation
        # Rolling std is tricky with convolution. E[X^2] - (E[X])^2

        # 1. Square data
        data_sq = data**2

        # 2. SMA of squared data
        sma_sq = self.moving_average(data_sq, window)

        # 3. Variance = SMA_sq - SMA^2
        var = sma_sq - sma**2
        std = torch.sqrt(
            torch.clamp(var, min=0)
        )  # clamp to avoid negative due to precision

        upper = sma + (std * num_std)
        lower = sma - (std * num_std)

        return upper, sma, lower
    def compute_imbalance(self, bid_vol: torch.Tensor, ask_vol: torch.Tensor) -> torch.Tensor:
        """
        Calculate Order Book Imbalance.
        (BidVol - AskVol) / (BidVol + AskVol)
        """
        bid_vol = self._to_tensor(bid_vol)
        ask_vol = self._to_tensor(ask_vol)
        return (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-10)

    def compute_spread(self, best_bid: torch.Tensor, best_ask: torch.Tensor) -> torch.Tensor:
        """Calculate Relative Spread: (Ask - Bid) / MidPrice."""
        best_bid = self._to_tensor(best_bid)
        best_ask = self._to_tensor(best_ask)
        mid = (best_bid + best_ask) / 2.0
        return (best_ask - best_bid) / (mid + 1e-10)

    def compute_vwap(self, prices: torch.Tensor, volumes: torch.Tensor, window: int = 20) -> torch.Tensor:
        """
        Calculate Volume-Weighted Average Price (VWAP) over a rolling window.
        """
        prices = self._to_tensor(prices)
        volumes = self._to_tensor(volumes)
        
        pv = prices * volumes
        
        sum_pv = self.moving_average(pv, window) * window # moving_average returns mean, so multiply by window
        sum_vol = self.moving_average(volumes, window) * window
        
        return sum_pv / (sum_vol + 1e-10)

    def compute_price_momentum(self, prices: torch.Tensor, window: int = 10) -> torch.Tensor:
        """Calculate price momentum (rate of change)."""
        prices = self._to_tensor(prices)
        return (prices - prices.roll(window)) / (prices.roll(window) + 1e-10)
