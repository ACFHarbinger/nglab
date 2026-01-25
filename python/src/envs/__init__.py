"""
nglab Arena - Python wrapper for Rust RL trading environment

This module provides Gymnasium-compatible environments for:
- CLOB (Central Limit Order Book) trading
- Polymarket prediction markets
- General trading simulation
"""

from __future__ import annotations

from .envs import ClobEnv, PolymarketEnv, TradingEnv

__all__ = ["ClobEnv", "PolymarketEnv", "TradingEnv"]
