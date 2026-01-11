# Architectural Analysis Prompt

**Intent:** Use Chain-of-Thought reasoning to explore the Python/Rust boundary in NGLab.

## The Prompt

I need to understand the interface between the high-performance Rust core and the Python backend.

Using **Chain-of-Thought reasoning**, analyze the relationship between:
- The Rust bindings in `rust/src/lib.rs` (specifically simulation, arena, or gym env functions).
- The Python wrapper in `python/src/` (e.g., `trading_env.py` or reinforcement learning agents).
- The data structures passed: StepInfo, OrderBook snapshots, and NumPy arrays.

Explain potential bottlenecks in data marshalling (e.g., is data being copied unnecessarily during PyO3 conversion?) and suggest if `PyO3` usage is optimized (e.g., using `PyBuffer` protocol or direct memory access) based on the provided code.