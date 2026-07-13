# Roadmap — Core Hub (Rust, Tier 0 / Control)

The Rust Tauri backend is the "air-traffic controller": it bridges data to the TypeScript
frontend via `#[tauri::command]`, owns prediction-market and EVM logic, and manages the lifecycle
of the Go and C++ daemons. Implementation lives in [`rust/`](../../rust/) and
[`typescript/src-tauri/`](../../typescript/src-tauri/).

## §1 — Retained native-Rust responsibilities (enforced)

- Prediction Market logic stays entirely in native Rust.
- EVM smart-contract monitoring via the **Alloy** crate stays in native Rust.
- State management and the `#[tauri::command]` bridge to the TS frontend stay in Rust.

## §2 — Binary lifecycle management (Priority 0)

- [ ] Spin up the Go Crypto Daemon with a chosen dynamic port
      (`crypto-daemon --port=<n>`) via `std::process::Command` or Tauri's Shell plugin.
- [ ] Spin up the C++ HFT daemon and attach to its shared-memory segment as a reader.
- [ ] Health-monitor both; restart on crash with backoff; propagate status to the UI.
- [ ] Clean shutdown: SIGTERM children, unmap shm, close loopback.

## §3 — De-scoping (migration)

- [ ] Remove crypto feed/exchange/JSON-RPC code paths (now in Go — [crypto_go.md](crypto_go.md)).
- [ ] Remove sub-µs execution / raw matching (now in C++ — [hft_cpp.md](hft_cpp.md)).
- [ ] Keep thin adapter layers: loopback reader (from Go), shm reader (from C++).

## §4 — IPC readers

- [ ] Loopback client for the Go bridge (length-prefixed Protobuf frames).
- [ ] Lock-free shm reader for the C++ metrics block (seqlock/double-buffer aware).
- [ ] Surface both as unified `#[tauri::command]` streams to the frontend.
