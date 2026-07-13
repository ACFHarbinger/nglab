# Roadmap — Strategy Brain (Python, Offline / Analytical)

Python stays an **offline analytical loop**: AI/ML models and quantitative strategies, isolated
from the live hot path. It processes database records and writes prediction weights to a
lightweight flat binary array or JSON config for instant hot-reloading by C++/Rust in live memory.
Implementation lives in the [`python/`](../../python/) submodule.

## §1 — Boundary (enforced)

- No live-trading execution in Python; it never sits on the tick-to-order path.
- Consumes historical/DB records; produces weights + configs only.

## §2 — Prediction-weight export

- [ ] Serialize trained model weights to a flat binary array (fixed layout, versioned header) for
      zero-parse `mmap` loading by the C++ HFT loop and the Rust hub.
- [ ] Also emit a JSON config for hot-reloadable, human-readable parameters.
- [ ] Define a stable schema version so consumers can validate on load.

## §3 — Research → production handoff

- [ ] Backtests read the same Protobuf `Tick`/`Order` records (see
      [schema_protobuf.md](schema_protobuf.md)) the live tiers emit.
- [ ] Document the hot-reload contract (file paths, atomic write-then-rename, versioning).
