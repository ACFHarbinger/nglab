# NGLab Field Repair Manual: Troubleshooting Guide

> **Diagnosis > Guesswork**.
> This document maps "Symptoms" to "Root Causes" and "Cures".

This comprehensive troubleshooting guide is organized by symptom category. Use the table of contents to quickly navigate to your issue.

---

## Table of Contents

1.  [Quick Diagnostics: The Health Check](#1-quick-diagnostics-the-health-check)
2.  [Environment Issues](#2-environment-issues)
3.  [Build Failures](#3-build-failures)
4.  [Runtime Crashes](#4-runtime-crashes)
5.  [Logic Errors](#5-logic-errors)
6.  [Performance Bottlenecks](#6-performance-bottlenecks)
7.  [Network & API Issues](#7-network--api-issues)
8.  [Database Issues](#8-database-issues)
9.  [GUI/Tauri Issues](#9-guitauri-issues)
10. [Docker & Deployment Issues](#10-docker--deployment-issues)
11. [Common Error Messages Reference](#11-common-error-messages-reference)
12. [Asking for Help](#12-asking-for-help)

---

## 1. Quick Diagnostics: The Health Check

Before diving deep, run the automated health check suite.

```bash
# Checks if Rust toolchain, Python venv, and Node modules are synced.
just check-health
```

**Expected Output:**
```text
[✅] Rust: cargo 1.80.0
[✅] Python: 3.11.4 (Active Venv: .venv)
[✅] Node: v20.5.1
[✅] Database: Connected (markets.db)
[✅] Rust Extension: nglab module importable
```

If any of these fail, proceed to the relevant section below.

### Quick Fix Commands

| Issue | Quick Fix |
|-------|-----------|
| Rust not found | `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \| sh` |
| Python venv missing | `uv sync` |
| Node modules missing | `cd typescript && npm ci` |
| Rust extension not built | `just build-python` |

---

## 2. Environment Issues

### 2.1 Rust / Cargo

#### Symptom: `error: linker 'cc' not found`
*   **Cause**: Missing system build essentials.
*   **Fix**:
    ```bash
    sudo apt update && sudo apt install build-essential pkg-config libssl-dev
    ```

#### Symptom: `PyO3: undefined symbol: _Py_NoneStruct`
*   **Context**: Running a Rust binary that depends on `pyo3` but isn't a Python extension module.
*   **Cause**: You cannot run `cargo run` on a `cdylib` crate meant for Python import.
*   **Fix**:
    *   Do NOT run `cargo run` for the library.
    *   Use `maturin develop` to build and install into your venv.
    *   Run the *Python entry point* that imports the Rust `.so`.

#### Symptom: `custom-build` Metadata Error
*   **Cause**: `protobuf-codegen` failing to find `protoc`.
*   **Fix**:
    ```bash
    sudo apt install protobuf-compiler
    ```

#### Symptom: `error[E0433]: failed to resolve: use of undeclared crate or module`
*   **Cause**: Missing dependency in `Cargo.toml` or workspace member not configured.
*   **Fix**:
    ```bash
    # Check if the crate is in workspace members
    grep -r "members" Cargo.toml
    # Add missing dependency
    cargo add <crate_name>
    ```

### 2.2 Python / UV

#### Symptom: `ModuleNotFoundError: No module named 'nglab'`
*   **Cause**: The Rust extension hasn't been built/installed into the current virtual environment.
*   **Fix**:
    ```bash
    source .venv/bin/activate
    just build-python # Runs maturin develop
    ```

#### Symptom: `RuntimeError: PyTorch not compiled with CUDA enabled`
*   **Cause**: You installed the CPU-only version of Torch.
*   **Fix**:
    ```bash
    uv pip install torch --index-url https://download.pytorch.org/whl/cu118 --force-reinstall
    ```

#### Symptom: `ImportError: libcudnn.so.8: cannot open shared object file`
*   **Cause**: CUDA libraries not in PATH.
*   **Fix**:
    ```bash
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
    # Add to ~/.bashrc for persistence
    ```

#### Symptom: `TypeError: 'NoneType' object is not subscriptable` during config loading
*   **Cause**: Hydra configuration file missing or malformed.
*   **Fix**:
    *   Check that `python/src/conf/config.yaml` exists.
    *   Validate YAML syntax: `python -c "import yaml; yaml.safe_load(open('config.yaml'))"`

### 2.3 Tauri / Frontend

#### Symptom: `WebView2Loader.dll not found` (Windows)
*   **Cause**: Missing webview runtime.
*   **Fix**: Install the "Evergreen Bootstrapper" from Microsoft.

#### Symptom: `EACCES: permission denied, access '/usr/lib/node_modules'`
*   **Cause**: You are running npm with global permissions improperly.
*   **Fix**: Do not use `sudo`. Use `nvm` to manage node versions.

#### Symptom: `Cannot find module 'typescript'`
*   **Cause**: Node modules not installed.
*   **Fix**:
    ```bash
    cd typescript
    rm -rf node_modules package-lock.json
    npm install
    ```

---

## 3. Build Failures

### 3.1 Rust Build Failures

#### Symptom: `error: could not compile 'nglab'`
*   **Diagnosis**:
    1.  Read the full error message—Rust errors are descriptive.
    2.  Check for missing system dependencies.
*   **Common Causes**:
    *   Missing `openssl-dev`: `sudo apt install libssl-dev`
    *   Missing `pkg-config`: `sudo apt install pkg-config`

#### Symptom: `error[E0308]: mismatched types` in PyO3 code
*   **Cause**: Rust-Python type mismatch.
*   **Fix**: Check that your `#[pyfunction]` return types match PyO3 expectations.

### 3.2 Python Build Failures

#### Symptom: `maturin develop` fails with `PEP 517` error
*   **Cause**: Virtual environment not activated or corrupted.
*   **Fix**:
    ```bash
    deactivate
    rm -rf .venv
    uv sync
    source .venv/bin/activate
    maturin develop
    ```

### 3.3 TypeScript Build Failures

#### Symptom: `TS2307: Cannot find module` for internal imports
*   **Cause**: TypeScript path aliases not configured.
*   **Fix**: Check `tsconfig.json` for correct `paths` configuration.

#### Symptom: `FATAL ERROR: Reached heap limit Allocation failed - JavaScript heap out of memory`
*   **Cause**: Node.js running out of memory during build.
*   **Fix**:
    ```bash
    export NODE_OPTIONS="--max-old-space-size=8192"
    npm run build
    ```

---

## 4. Runtime Crashes

### 4.1 Rust Panics (Core Dump)

**Symptom**: The terminal says `thread 'main' panicked at '...'`.

**Action**: Backtrace it.
```bash
# Run with backtrace enabled
RUST_BACKTRACE=1 cargo run --bin nglab-cli
```

**Common Panics**:

| Panic Message | Cause | Fix |
|---------------|-------|-----|
| `index out of bounds` | Accessing beyond array length | Check loop bounds, use `.get()` instead of `[]` |
| `called Option::unwrap() on a None value` | Unwrapping empty Option | Use pattern matching or `if let` |
| `borrow error` | RefCell double borrow | Reduce scope of first borrow |
| `arithmetic overflow` | Integer overflow | Use `checked_add()` or `saturating_add()` |

### 4.2 Python Segfaults (Segmentation Fault)

**Symptom**: Process exits with `Exit Code 139` with no Python traceback.

**Root Cause**: Typically undefined behavior in the Rust `unsafe` block or invalid pointer access in PyO3.

**Action**: Debug with GDB/LLDB.
```bash
gdb --args python python/src/main.py
(gdb) run
# ... Wait for crash ...
(gdb) bt
```
Look for the top frame. If it's inside `libnglab.so`, it's a Rust bug. File a critical issue.

### 4.3 Python Exceptions

**Symptom**: Standard Python traceback.

**Diagnosis**:
1.  Read the full traceback from bottom to top.
2.  The last frame is where the error occurred.
3.  Check the error type (KeyError, ValueError, etc.).

---

## 5. Logic Errors

### 5.1 The Agent isn't learning (Reward stays flat)

**Diagnosis Checklist**:

1.  **Check Normalization**: Are inputs normalized to $\mathcal{N}(0,1)$? Raw prices (e.g., 20,000) break neural net gradients.
    *   *Fix*: Ensure `ObservationNormalizer` is active in `gym.rs`.
2.  **Check Rewards**: Is the reward signal strictly zero?
    *   *Fix*: Print `reward` at every step. Ensure `transaction_cost` isn't eating all profits.
3.  **Check Entropy**: If entropy drops to 0 immediately, the Learning Rate is too high.
    *   *Fix*: Reduce LR by 10x.
4.  **Check Gradient Flow**: Are gradients NaN or zero?
    *   *Fix*: Use gradient clipping and check for division by zero.

### 5.2 Order Book Desync

**Symptom**: Frontend shows a price of $100, but Agent buys at $101.

**Cause**: Physics/Rendering Latency mismatch.
*   The Rust engine operates at 10kHz.
*   The UI updates at 60Hz.
*   The UI is showing "old news".

**Fix**: Trust the logs, not the UI. The UI is a downsampled approximation of the truth.

### 5.3 Model Predicts Constant Values

**Symptom**: The model always outputs the same action/prediction.

**Causes**:
1.  **Dead ReLU**: All neurons have become inactive.
2.  **Collapsed VAE**: KL divergence dominating, latent space unused.
3.  **Mode Collapse in GAN**: Generator found a single "safe" output.

**Fixes**:
1.  Use LeakyReLU instead of ReLU.
2.  Implement KL annealing in VAE training.
3.  Use spectral normalization in GAN.

### 5.4 Overfitting to Training Data

**Symptom**: Excellent training metrics, poor validation/test metrics.

**Fixes**:
1.  Add dropout layers.
2.  Use data augmentation.
3.  Reduce model capacity (fewer layers/parameters).
4.  Increase regularization (weight decay).

---

## 6. Performance Bottlenecks

### 6.1 Slow Step Times (>10ms)

**Action**: Profile it.
```bash
# Install samply
cargo install samply

# Profile the execution
samply record python python/src/main.py
```
View the flamegraph.
*   If `PyO3::to_py_object` is huge: You are copying data. Switch to `numpy` views.
*   If `Mutex::lock` is huge: You have high thread contention between the GUI reader and the simulation writer. Use `RwLock` or double-buffering.

### 6.2 GPU Starvation (0% Util)

**Cause**: The CPU isn't feeding data fast enough.

**Fix**:
*   Increase `num_envs` (Vectorized Environments).
*   Move `ObservationNormalizer` to Rust (prevent Python GIL lock).
*   Use async data loading with `num_workers > 0` in DataLoader.

### 6.3 Memory Leaks

**Symptom**: Memory usage grows continuously over time.

**Diagnosis**:
```bash
# Python memory profiling
pip install memory-profiler
python -m memory_profiler python/src/main.py
```

**Common Causes**:
1.  Storing all history in a list that never gets cleared.
2.  Circular references preventing garbage collection.
3.  Large tensors not being freed (use `del tensor; torch.cuda.empty_cache()`).

### 6.4 Slow Training

**Symptom**: Training takes hours/days when it should take minutes/hours.

**Checklist**:
1.  Is GPU being used? Check `nvidia-smi`.
2.  Is data loading the bottleneck? Check if GPU util is <50%.
3.  Is the model too large? Profile forward/backward pass times.

---

## 7. Network & API Issues

### 7.1 Polymarket API Errors

#### Symptom: `HTTPError: 429 Too Many Requests`
*   **Cause**: Rate limiting.
*   **Fix**: Implement exponential backoff:
    ```python
    import time
    for i in range(5):
        try:
            response = api.request()
            break
        except RateLimitError:
            time.sleep(2 ** i)
    ```

#### Symptom: `ConnectionError: Connection refused`
*   **Cause**: API server down or network issue.
*   **Fix**: Check internet connection; retry later.

### 7.2 WebSocket Disconnections

**Symptom**: Stream stops receiving updates.

**Fix**: Implement reconnection logic with exponential backoff.

---

## 8. Database Issues

### 8.1 SQLite Locked

#### Symptom: `sqlite3.OperationalError: database is locked`
*   **Cause**: Multiple instances of the scraper or backend accessing `markets.db`.
*   **Fix**: 
    *   Ensure only one process writes to the database.
    *   Use `just reset-credentials` to clear local databases if they become corrupted.

### 8.2 Database Corruption

**Symptom**: `database disk image is malformed`

**Fix**:
```bash
# Backup corrupted database
mv markets.db markets.db.bak

# Create fresh database
python -c "from nglab.db import init_db; init_db()"
```

---

## 9. GUI/Tauri Issues

### 9.1 Blank Screen on Launch

**Symptom**: Tauri window opens but shows white/blank screen.

**Causes**:
1.  Frontend build failed.
2.  Incorrect asset paths.

**Fix**:
```bash
cd typescript
npm run build
npm run tauri dev
```

### 9.2 "arena-update" events not firing

*   **Cause**: The Rust backend loop is not started or the Mutex is deadlocked.
*   **Fix**: Check `src-tauri/src/main.rs` for Tokio task logs. Ensure `ArenaState` is correctly unlocked in and outside the loop.

### 9.3 Charts Not Rendering

**Symptom**: Chart containers exist but no data appears.

**Causes**:
1.  Data format mismatch.
2.  Chart library not initialized.

**Fix**: Check browser console for JavaScript errors. Verify data shape matches chart expectations.

---

## 10. Docker & Deployment Issues

### 10.1 Docker Build Failures

#### Symptom: `COPY failed: file not found`
*   **Cause**: File path in Dockerfile is relative to wrong context.
*   **Fix**: Ensure `.dockerignore` isn't excluding required files.

#### Symptom: `OCI runtime create failed: container_linux.go:xxx: starting container process caused...`
*   **Cause**: Binary incompatibility (e.g., built for wrong architecture).
*   **Fix**: Ensure multi-arch build is configured correctly.

### 10.2 Container Crashes on Startup

**Diagnosis**:
```bash
docker logs <container_id>
```

**Common Causes**:
1.  Missing environment variables.
2.  Port already in use.
3.  Insufficient memory.

### 10.3 Kubernetes Deployment Issues

#### Symptom: Pod in `CrashLoopBackOff`
*   **Diagnosis**: `kubectl logs <pod-name>` and `kubectl describe pod <pod-name>`
*   **Common Fixes**:
    *   Check resource limits (memory, CPU).
    *   Verify secrets and configmaps are mounted.

---

## 11. Common Error Messages Reference

| Error Message | Likely Cause | Quick Fix |
|---------------|--------------|-----------|
| `ModuleNotFoundError: No module named 'nglab'` | Rust extension not built | `just build-python` |
| `error: linker 'cc' not found` | Missing build tools | `sudo apt install build-essential` |
| `CUDA out of memory` | Batch size too large | Reduce `batch_size` in config |
| `RuntimeError: CUDA error: device-side assert triggered` | NaN in computation | Check for division by zero |
| `OSError: [Errno 28] No space left on device` | Disk full | Free up disk space, check `/tmp` |
| `TimeoutError` | Network or CPU timeout | Increase timeout, check connectivity |
| `PermissionError` | File access denied | Check file permissions, avoid `sudo` |

---

## 12. Asking for Help

When opening an issue, provide the **"Crash Tuple"**:

1.  **The Command**: Exactly what you typed.
2.  **The Stack Trace**: The full output (use `RUST_BACKTRACE=1`).
3.  **The Context**: Commit hash (`git rev-parse HEAD`) and OS.
4.  **The Reproduction Steps**: Minimal steps to reproduce the issue.
5.  **Expected vs Actual Behavior**: What you expected to happen and what actually happened.

### Issue Template

```markdown
## Environment
- OS: Ubuntu 22.04
- Rust: 1.80.0
- Python: 3.11.4
- Commit: abc1234

## Steps to Reproduce
1. Run `just setup`
2. Run `just dev`
3. Click "Start Arena"

## Expected Behavior
Arena should start and emit events.

## Actual Behavior
Application crashes with the following error:
```error message here```

## Full Stack Trace
```paste here```
```

**Emergency Contacts**:
*   **Infrastructure**: @devops-lead
*   **Rust Core**: @rust-ace
*   **Models**: @ml-researcher
*   **Frontend**: @frontend-dev

---

**Remember**: The best debugging is prevention. Write tests, use type hints, and read the documentation.
