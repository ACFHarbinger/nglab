# Pull Request

## Summary

<!-- What does this PR change and why? Link the roadmap item (moon/ROADMAP.md or a module roadmap in moon/roadmaps/). -->

## Affected Tier(s)

- [ ] Core Hub (Rust / Tauri)
- [ ] Crypto Daemon (Go)
- [ ] HFT Native Loop (C++)
- [ ] Strategy Brain (Python / ML)
- [ ] Control Panel (TypeScript / React)
- [ ] Universal Schema (Protobuf)
- [ ] Tooling / docs / CI

## Type of Change

- [ ] 🐛 Bug fix
- [ ] ✨ New feature
- [ ] ♻️ Refactor / migration (Rust → Go/C++)
- [ ] ⚡ Performance / latency
- [ ] 📚 Documentation
- [ ] 🔧 Tooling / CI

## Language-Boundary Checklist

- [ ] Crypto market-data / exchange / concurrent-feed logic is in **Go** (not Rust/C++).
- [ ] Ultra-low-latency / HFT-venue execution is in **C++** (not Rust/Go).
- [ ] Prediction markets + EVM (Alloy) monitoring remain **native Rust**.
- [ ] AI/ML stays in **Python** (offline analytical loop; no live hot-path logic).
- [ ] TypeScript stays **thin** (stream consumer + execution triggers; no business logic).
- [ ] Cross-boundary structs use the **Protobuf** universal schema (regenerated for all langs).
- [ ] IPC respects the transport rule: **Go↔Rust loopback**, **C++↔Rust shared memory** (no sockets for C++).

## Verification

- [ ] `just lint` and `just test` pass for the affected tier(s).
- [ ] Benchmarks / latency numbers included for hot-path (C++) changes.
- [ ] Docs / roadmaps / CHANGELOG updated where the public surface changed.
