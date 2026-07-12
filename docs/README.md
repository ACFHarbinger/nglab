# NGLab Documentation

Central index for NGLab's documentation. The master roadmap and per-module roadmaps live under
[`moon/`](../moon/); contribution and coverage config under [`git/`](../git/).

## Guides

| Document | Purpose |
| :--- | :--- |
| [ARCHITECTURE.md](ARCHITECTURE.md) | System blueprint: tiers, boundaries, data flow, IPC |
| [DEVELOPMENT.md](DEVELOPMENT.md) | Environment setup, IDE config, local dev, profiling |
| [TESTING.md](TESTING.md) | Testing philosophy, coverage, mocking, CI |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | Common issues and diagnostics |
| [TUTORIAL.md](TUTORIAL.md) | Developer encyclopedia — per-module deep dives |
| [DEPENDENCIES.md](DEPENDENCIES.md) | Backend, frontend, and ML dependency inventory |
| [IMPROVEMENT_GUIDE.md](IMPROVEMENT_GUIDE.md) | Refactoring and improvement playbook |
| [CHANGELOG.md](CHANGELOG.md) | Release history (Keep a Changelog) |

## Architecture Decision Records (ADRs)

Numbered ADRs (`0001`–`0012`, plus [`template.md`](template.md)) record the key technology choices —
Rust for simulation, PyO3 bindings, Tauri frontend, Gymnasium interface, SQLCipher, OpenTelemetry,
and more.

## Generated API references

`just docs` builds language API docs into this tree (`docs/rust/`, `docs/python/` — git-ignored).

## Roadmaps

See [`moon/ROADMAP.md`](../moon/ROADMAP.md) (master) and [`moon/roadmaps/`](../moon/roadmaps/) for
the polyglot module roadmaps (Go crypto, C++ HFT, Rust core, Python strategy, TS frontend,
Protobuf schema).
