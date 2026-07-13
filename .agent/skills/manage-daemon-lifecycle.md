# Skill: Wire a Daemon into the Rust Lifecycle Manager

The Rust core hub owns the Go and C++ binaries. Use this when adding/adjusting a managed daemon.

1. **Spawn**: launch via `std::process::Command` (or Tauri's Shell plugin). For the Go crypto
   daemon, pick a free dynamic port and pass `--port=<n>` (loopback, avoids privileged-bind
   permission errors). For the C++ HFT daemon, pass the shared-memory segment name/size.
2. **Attach**: Go → connect the loopback client (length-prefixed Protobuf frames). C++ → `mmap`
   the shm segment as a **reader** (lock-free; seqlock/double-buffer aware — never write).
3. **Monitor**: health-check/heartbeat; restart with backoff on exit; surface status to the UI via
   `#[tauri::command]`/events.
4. **Shutdown**: SIGTERM children, join, unmap shm, close loopback — no orphans.
5. **Boundary**: keep this in Rust; the daemons themselves stay Go/C++. Update
   `moon/roadmaps/core_rust.md`.
