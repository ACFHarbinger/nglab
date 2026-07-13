# Skill: Migrate Logic Out of Rust (→ Go or C++)

Refactor existing Rust functionality into the tier where it now belongs.

1. **Classify**: crypto market-data / exchange / concurrent-feed logic → **Go** (`go/`);
   sub-µs execution / raw matching / HFT-venue logic → **C++** (`cpp/`). Prediction markets and
   EVM (Alloy) stay in Rust.
2. **Inventory**: list the Rust modules/functions to move; note their public boundaries and any
   PyO3/ts-rs exposure.
3. **Port**: reimplement in the target language following its skill
   ([add-crypto-feed](add-crypto-feed.md) / [add-hft-strategy](add-hft-strategy.md)); route data
   across the correct IPC (Go→loopback, C++→shared memory).
4. **Bridge in Rust**: replace the moved logic with a thin reader/adapter; wire lifecycle
   (spawn/monitor/restart the new binary) via `std::process::Command` or the Tauri Shell plugin.
5. **Verify parity**: same inputs → same outputs; for C++ also verify latency ≤ the Rust baseline.
6. **Remove** the old Rust path once parity holds. Update the relevant roadmap + CHANGELOG.
