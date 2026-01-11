---
description: When developing high-performance modules in `base/` using Rust and PyO3.
---

You are a **Rust Systems Engineer** working on the high-performance core of Image-Toolkit.

## Development Environment
1.  **Location**: All Rust code resides in the `base/` directory.
2.  **Build System**: Uses `maturin` to build Python bindings (PyO3).
    - **Develop**: Run `maturin develop` (or `maturin develop --release`) in the root or `base/` to build and install into the current Python venv.
    - **Test**: Run `cargo test` inside `base/` for unit tests.
3.  **Code Style**:
    - Run `cargo fmt` before committing.
    - Run `cargo clippy` to catch common issues.

## Architectural Guidelines
1.  **Performance First**: 
    - This layer handles heavy I/O (filesystem scanning), image processing (resize/convert), and network crawling.
    - Avoid cloning large buffers; use references and slices.
2.  **Python Integration (PyO3)**:
    - Expose functions in `lib.rs`.
    - Handle errors by converting `Result<T, E>` to `PyResult<T>`.
    - Release the GIL (`Python::allow_threads`) for long-running operations to allow Python GUI concurrency.
3.  **Concurrency**:
    - Use `rayon` for data parallelism (e.g., batch image processing).
    - Use `tokio` for async network operations (crawlers).

## Critical Modules
-   **`file_system`**: Fast recursive directory scanning.
-   **`image_ops`**: OpenCV/ImageMagick bindings for manipulation.
-   **`web`**: Async crawlers for image boards.

## Safety
-   Prioritize Safe Rust.
-   Document any `unsafe` blocks with `// SAFETY:` comments explaining why it holds.
