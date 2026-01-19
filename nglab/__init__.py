"""
nglab - Next Gen Laboratory

Multimodal Deep Reinforcement Learning for financial trading.
"""

# Export Rust bindings
try:
    from ._nglab import Arena, OrderBook, PolymarketArena, TradingEnv

    __all__ = ["Arena", "OrderBook", "PolymarketArena", "TradingEnv"]
except ImportError:
    # Allow importing without compiled bindings (e.g. for docs or light usage)
    pass
