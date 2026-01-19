# NGLab Troubleshooting Guide

This guide covers common issues and resolutions for the NGLab development environment.

## 1. Rust Simulation Engine

### Simulation Panic: "assertion failed: left == right"
- **Cause**: Usually related to floating-point precision in order book tests or mismatched trade side logic in `MultiAssetEnv`.
- **Solution**: Ensure usage of `.round()` or `f64::EPSILON` for comparisons. Verify that `Trade` side references are correctly interpreted as either Maker or Taker side (standard is Taker).

### PyO3: "undefined symbol: _Py_NoneStruct"
- **Cause**: Occurs when running standalone Rust binaries that depend on Python features without linking to the Python interpreter.
- **Solution**: Ensure you are using `maturin develop` to build the Python bindings, or run `cargo test --no-default-features` if the `python` feature is causing issues.

## 2. Python Environment

### ModuleNotFoundError: 'nglab'
- **Cause**: The Rust package hasn't been compiled into the Python environment.
- **Solution**: Run `just build-python` or `cd python && maturin develop`.

### Protocol Error in MultiAssetEnv Observation
- **Cause**: The number of assets or features per asset doesn't match the expected `ndarray` shape in the Python wrapper.
- **Solution**: Verify `features_per_asset` in `rust/src/simulation/multi_asset.rs` matches the shape defined in the Python `step` wrapper.

## 3. Frontend / Tauri

### Webview fails to load
- **Cause**: Permissions or system dependencies (especially on Linux).
- **Solution**: Run `sudo apt install libwebkit2gtk-4.0-dev build-essential curl wget libssl-dev libgtk-3-dev libayatana-appindicator3-dev librsvg2-dev`.

### "arena-update" events not firing
- **Cause**: The Rust backend loop is not started or the Mutex is deadlocked.
- **Solution**: Check `src-tauri/src/main.rs` for Tokio task logs. Ensure `ArenaState` is correctly unlocked in and outside the loop.

## 4. General DX

### "Just" command not found
- **Solution**: Install it via `cargo install just` or `sudo apt install just`.

### Database Locked (SQLite)
- **Cause**: Multiple instances of the scraper or backend accessing `markets.db`.
- **Solution**: Use `just reset-credentials` to clear local databases if they become corrupted.
