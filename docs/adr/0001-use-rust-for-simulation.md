# ADR-0001: Use Rust for Market Simulation and Core Logic

## Status
Accepted

## Context
Market simulation requires high performance to handle millions of order book updates and trade events with minimal latency. Python, while excellent for ML and data science, often becomes a bottleneck for low-level simulation logic and multi-threaded execution.

## Decision
We will implement the core simulation engine, including the `OrderBook`, `Arena`, and `TradingEnv` logic, in Rust. Rust provides C-level performance, memory safety, and excellent concurrency primitives (Tokio) without garbage collection overhead.

## Consequences
- **Easier**: Higher simulation speeds, thread safety guaranteed by the compiler, and efficient resource utilization.
- **Difficult**: Higher barrier to entry for developers only familiar with Python. Requires complex foreign function interface (FFI) bindings via PyO3.

## Alternatives Considered
- **Pure Python**: Too slow for high-fidelity LOB simulation.
- **C++**: Good performance, but lacks the safety guarantees and modern ecosystem/tooling that Rust offers.
- **Julia**: Strong performance, but integration with the Python/Tauri ecosystem is less mature than Rust's.
