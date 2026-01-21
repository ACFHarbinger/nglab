# ADR-0003: Use PyO3 for Python Bindings

## Status
Accepted

## Context
The project architecture splits responsibilities between a high-performance simulation engine (Rust) and a flexible machine learning/scripting layer (Python). We need a mechanism to bridge these two worlds efficiently, allowing Python code to leverage Rust structs and functions with minimal overhead.

## Decision
We will use **PyO3** to create Python bindings for our Rust code. Specifically:
- Core data structures (e.g., `Arena`, `OrderBook`, `MultiAssetEnv`) will be exposed as Python classes using `#[pyclass]`.
- Performance-critical functions will be exposed using `#[pyfunction]`.
- We will use `maturin` as the build and publication tool to manage the complexities of compiling Rust extensions for Python.

## Consequences
- **Easier**:
    - "Zero-copy" access to Rust memory from Python in many cases.
    - Automatic type conversion between Rust and Python types.
    - Ability to write Python-native extensions purely in Rust.
- **Difficult**:
    - Debugging across the language boundary can be complex (e.g., analyzing core dumps).
    - The build process becomes more involved, requiring both `cargo` and Python build tools.
    - Developers must manage the Global Interpreter Lock (GIL) explicitly when multi-threading in Rust.

## Alternatives Considered
- **ctypes / CFFI**: Would require exposing a C-ABI from Rust (`extern "C"`), which is unsafe and requires manual memory management conformant to C standards.
- **rust-cpython**: An alternative to PyO3, but PyO3 has a larger community, better documentation, and actively maintained `maturin` integration.
- **IPC / GRPC**: Running Rust as a separate service communicating over sockets. This would introduce significant serialization/deserialization latency, which is unacceptable for the high-frequency interaction required during RL training.
