"""
nglab Arena - Python wrapper for Rust RL trading environment

This module provides Gymnasium-compatible environments for:
- CLOB (Central Limit Order Book) trading
- Polymarket prediction markets
- General trading simulation
"""

from .envs import ClobEnv, PolymarketEnv, TradingEnv

__all__ = ["ClobEnv", "PolymarketEnv", "TradingEnv"]
__version__ = "0.1.0"
