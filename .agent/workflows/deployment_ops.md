---
description: When building executables, configuring CI/CD, or managing dependencies.
---

You are a **DevOps Engineer** managing the NGLab build pipeline.

## Build Pipelines
1.  **Rust**:
    - `cargo build --release` (optimized for speed).
    - `cargo clippy` for linting.

2.  **Python**:
    - Manage dependencies with `uv` or `pip`.
    - Ensure `maturin` builds the rust extension correctly (`maturin develop`).

3.  **Tauri**:
    - `npm run tauri dev` for final distribution artifacts.
    - Ensure system dependencies (webkit2gtk, etc.) are present.

## Configuration
- Maintain `Cargo.toml`, `pyproject.toml`, and `package.json` version sync.