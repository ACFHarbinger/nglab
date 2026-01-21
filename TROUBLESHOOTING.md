# NGLab Field Repair Manual: Troubleshooting Guide

> **Diagnosis > Guesswork**.
> This document maps "Symptoms" to "Root Causes" and "Cures".

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
```

If any of these fail, proceed to [Section 2: Environment Issues](#2-environment-issues).

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
    *   Use `maturin develop` to build and install into your key.
    *   Run the *Python entry point* that imports the Rust `.so`.

#### Symptom: `custom-build` Metadata Error
*   **Cause**: `protobuf-codegen` failing to find `protoc`.
*   **Fix**:
    ```bash
    sudo apt install protobuf-compiler
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

### 2.3 Tauri / Frontend

#### Symptom: `WebView2Loader.dll not found` (Windows)
*   **Cause**: Missing webview runtime.
*   **Fix**: Install the "Evergreen Bootstrapper" from Microsoft.

#### Symptom: `EACCES: permission denied, acces '/usr/lib/node_modules'`
*   **Cause**: You are running npm with global permissions improperly.
*   **Fix**: Do not use `sudo`. Use `nvm` to manage node versions.

---

## 3. Runtime Crashes (The "Panic" Room)

### 3.1 Rust Panics (Core Dump)
**Symptom**: The terminal says `thread 'main' panicked at '...'`.

**Action**: Backtrace it.
```bash
# Run with backtrace enabled
RUST_BACKTRACE=1 cargo run --bin nglab-cli
```

**Common Panics**:
1.  `index out of bounds`: You are accessing `history[200]` on a 200-len vec.
2.  `borrow error`: You are trying to `borrow_mut()` a `RefCell` that is already borrowed. **Fix**: Reduce scope of the first borrow.

### 3.2 Python Segfaults (Segmentation Fault)
**Symptom**: Process assumes `Exit Code 139` with no python traceback.

**Root Caue**: Typically undefined behavior in the Rust `unsafe` block or invalid pointer access in PyO3.

**Action**: Debug with GDB/LLDB.
```bash
gdb --args python python/src/main.py
(gdb) run
# ... Wait for crash ...
(gdb) bt
```
Look for the top frame. If it's inside `libnglab.so`, it's a Rust bug. File a critical issue.

---

## 4. Logic Errors (The "Why is it doing that?" Room)

### 4.1 The Agent isn't learning (Reward stays flat)
**Diagnosis Checklist**:
1.  **Check Normalization**: Are inputs normalized to $\mathcal{N}(0,1)$? Raw prices (e.g., 20,000) breaks neural net gradients.
    *   *Fix*: Ensure `ObservationNormalizer` is active in `gym.rs`.
2.  **Check Rewards**: Is the reward signal strictly zero?
    *   *Fix*: Print `reward` at every step. Ensure `transaction_cost` isn't eating all profits.
3.  **Check Entropy**: If entropy drops to 0 immediately, the Learning Rate is too high.

### 4.2 Order Book Desync
**Symptom**: Frontend shows a price of $100, but Agent buys at $101.

**Cause**: Physics/Rendering Latency mismatch.
*   The Rust engine operates at 10kHz.
*   The UI updates at 60Hz.
*   The UI is showing "old news".

**Fix**: Trust the logs, not the UI. The UI is a downsampled approximation of the truth.

---

## 5. Performance Bottlenecks

### 5.1 Slow Step Times (>10ms)
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

### 5.2 GPU Starvation (0% Util)
**Cause**: The CPU isn't feeding data fast enough.
**Fix**:
*   Increase `num_envs` (Vectorized Environments).
*   Move `ObservationNormalizer` to Rust (prevent Python GIL lock).

---

## 6. Asking for Help

When opening an issue, provide the **"Crash Tuple"**:

1.  **The Command**: Exactly what you typed.
2.  **The Stack Trace**: The full output (use `RUST_BACKTRACE=1`).
3.  **The Context**: Commit hash (`git rev-parse HEAD`) and OS.

**Emergency Contacts**:
*   **Infrastructure**: @devops-lead
*   **Rust Core**: @rust-ace
*   **Models**: @ml-researcher
