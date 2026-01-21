# ADR-0008: Centralized Error Handling

## Status
Accepted

## Context
Codebases often suffer from ad-hoc error handling, where "stringly typed" errors (`Box<dyn Error>`) or loose `anyhow` contexts make it difficult for caller code to handle specific failure modes programmatically. Without a structured error hierarchy, upper layers (like the UI or Python bindings) cannot easily distinguish between a "Network Error" (retryable) and a "Validation Error" (fatal).

## Decision
We will implement a **Centralized Error Handling** strategy using the `thiserror` crate.
- **`ArenaError`**: A single, comprehensive enum in `rust/src/errors/mod.rs` will define all domain errors.
- **Conversion**: We will implement `From<T>` traits to automatically wrap upstream errors (IO, CSV, JSON, Network) into `ArenaError` variants.
- **PyO3 Integration**: `ArenaError` will implement conversion to `pyo3::PyErr`, ensuring that Rust errors raise appropriate Python exceptions.

## Consequences
- **Easier**:
    - **Reliability**: The compiler forces us to handle or propagate every error path.
    - **Client Logic**: The frontend can react to specific error kinds (e.g., showing a login prompt on `AuthError`).
    - **Consistency**: Error messages follow a standard format via the `#[error("...")]` macro.
- **Difficult**:
    - **Boilerplate**: The `ArenaError` enum can grow large.
    - **Maintenance**: Adding a new error source requires modifying the central enum, potentially causing recompilation of dependent modules.

## Alternatives Considered
- **`anyhow` everywhere**: Excellent for applications (CLIs), but bad for libraries because it erases type information. Since our Rust core is a library for Python/Tauri, we need typed errors.
- **Module-specific errors**: Defining `OrderBookError`, `NetworkError`, etc. separately. This reduces coupling but makes the top-level `Result` type signature complex and harder to map uniformly to Python.
