# Roadmap — Code Quality & Human Understanding

Make the NGLab codebase self-documenting, well-architected, and approachable (a new contributor
productive within ~30 minutes). Derived from the former `docs/IMPROVEMENT_GUIDE.md` audit and
rephrased as an actionable, phased roadmap. Cross-cuts every tier
([core_rust](core_rust.md), [strategy_python](strategy_python.md),
[frontend_typescript](frontend_typescript.md)).

Completed items move to [`docs/CHANGELOG.md`](../../docs/CHANGELOG.md).

---

## §1 — Documentation

### Rust (critical) — ~100 `missing_docs` warnings from `cargo doc`

- [ ] `simulation/` (13 files): module-level `//!` docs + all public APIs.
- [ ] `execution/` (6): algorithm docs with complexity analysis.
- [ ] `web/exchanges/` (5): trait documentation + integration guides.
- [ ] `models/` (6): ML inference pipeline documentation.
- [ ] `security/` (4): security-model explanation.
- [ ] Enforce `#![warn(missing_docs)]` in `lib.rs` **in CI** (attribute already present).

### Python

- [ ] Google-style docstrings for all public APIs (focus `models/`, `envs/`, `pipeline/`).
- [ ] Regenerate `_nglab.pyi` to match the current Rust bindings exactly (add an "auto-generated" header).
- [ ] Document dataclass configs with `#:` comments (`configs/`).
- [ ] Type hints on all public functions (`from __future__ import annotations`).

### TypeScript

- [ ] JSDoc for all exported components (priority: hooks, complex components).
- [ ] Document props interfaces with `@param` (focus `terminal/`, `charts/`).
- [ ] `@example` blocks for complex hooks (`useArena`, `usePolymarket`, `useStreaming`).
- [ ] A `README.md` per feature folder.

### Visual diagrams (Mermaid, in `docs/`)

- [ ] System architecture (Rust ↔ Python ↔ TypeScript).
- [ ] Data flow (Order → Arena → CLOB → Fill).
- [ ] Training pipeline (Gym → Policy → Training Loop).
- [ ] Streaming architecture (WebSocket → Events → UI).

## §2 — Naming & Code Clarity

- [ ] Rename `rust/src/moon/` → `scrapers/` (or `data_ingestion/`) — cryptic name, high impact.
- [ ] Split `MultiAssetEnv` → `MultiAssetEnv` + `PortfolioManager` (overloaded responsibility).
- [ ] `ArenaUpdate` → `SimulationState`/`StepResult`; `AlgoOrderWidget` → `AlgorithmicOrderWidget`.
- [ ] Extract scattered magic numbers to named constants.
- [ ] Add "why" inline comments to the high-value algorithms: `orderbook.rs` (price-time priority),
      `gym.rs` (reward/observation), `features.rs` (feature engineering), `streaming.rs` (reconnect/backoff).

## §3 — Architecture

### Rust

- [ ] Extract a pure **`domain/`** layer (`orderbook`, `portfolio`, `position`, `types`) out of
      `simulation/`; keep `simulation/` as environment orchestration only.
- [ ] Add an **`infrastructure/`** layer (`exchanges/`, `storage/`, `streaming/`).
- [ ] Trait-based exchange abstraction (`Exchange`, `DataProvider`) replacing scattered concretes.
- [ ] Layered error hierarchy: `domain` / `infrastructure` / `application` errors with `From` impls.

> Note: some of these (crypto exchanges, streaming) intersect the polyglot migration — new
> exchange/feed work belongs in **Go** ([crypto_go.md](crypto_go.md)); reconcile before moving code.

### Python

- [ ] Consolidate the scattered factories into a unified registry with auto-discovery
      (`@register("model")` + import-time discovery).
- [ ] Feature flags module (`src/features/__init__.py`: `USE_RUST_ENV`, `ENABLE_PROFILING`, …).

### TypeScript

- [ ] Consolidate overlapping hooks (`useArena`/`usePolymarket`/`useStreaming`) into a Zustand
      store with slices (`stores/tradingStore.ts`, `persist` + `devtools`).
- [ ] Feature-based component organization (`components/features/*`, `shared/`, `layouts/`).

## §4 — Code Cleanup

- [ ] Rust: fix unused `slice_qty` (`execution/pov.rs`), `remaining_qty` (`execution/twap.rs`),
      unnecessary `mut` (`simulation/paper_trading.rs`), dead `symbol`/`instrument_name`
      (`web/exchanges/{binance,deribit}.rs`).
- [ ] TypeScript: remove unused imports; extract magic numbers to `constants.ts`; consolidate mock
      generators into `__mocks__/`; unify button styling into a `shared/Button`.

## §5 — Developer Experience

- [ ] pre-commit: add Python/TS checks (currently partial).
- [ ] Add `cargo-deny` (dep audit), `typedoc` (TS docs), `cargo tarpaulin` (Rust coverage) to CI.
- [ ] Commit a recommended `.vscode/settings.json` (rust-analyzer clippy, ruff, prettier, format-on-save).
- [ ] Onboarding: `QUICKSTART.md` (5-min setup), `just setup`, `GLOSSARY.md` (CLOB, LOB, slippage,
      Sharpe…), documented `.env.example`, common debugging scenarios in `docs/TROUBLESHOOTING.md`.

## §6 — Testing & Validation

- [ ] Coverage targets: Rust 60→80% (proptest for orderbook), Python 55→70% (fixtures + GPU mocking),
      TypeScript 30→60% (React Testing Library + MSW).
- [ ] Integration scenarios: order flow (submit→match→fill→position), training loop
      (reset→step→collect→update), streaming (connect→subscribe→data→reconnect).
- [ ] Criterion benchmarks for order matching (`benches/orderbook_bench.rs`); wire into CI.

## §7 — Phased Execution

| Phase | Duration | Items |
| :--- | :--- | :--- |
| **1 — Quick wins** | 1–2 days | Rust `missing_docs` in critical modules; remove unused imports/vars; `.vscode/settings.json`; `GLOSSARY.md`; `_nglab.pyi` header |
| **2 — Documentation sprint** | ~1 week | Full JSDoc; Python docstrings; Mermaid diagrams in `docs/`; 2–3 new ADRs |
| **3 — Architectural refactor** | 2–3 weeks | Rename `moon/`→`scrapers/`; extract Rust domain layer; consolidate Python registries; Zustand store |
| **4 — Testing & validation** | 1–2 weeks | Property tests for orderbook; E2E integration; perf benchmarking in CI; hit coverage targets |

## §8 — Success Metrics

| Metric | Baseline | Target |
| :--- | :--- | :--- |
| Rust doc warnings | ~100 | 0 |
| TypeScript lint warnings | ~20 | 0 |
| Python docstring coverage | ~40% | 90% |
| ADR count | 12 | 15+ |
| Test coverage (overall) | ~50% | 75% |
| Build time | ~60s | <45s |
| Onboarding time | unknown | <30 min |
| Avg code complexity | unknown | <10 |

## Appendix — Docstring templates

Rust `///` (Arguments/Returns/Example), Python Google-style (Args/Returns/Example), and TypeScript
JSDoc (`@param`/`@returns`/`@example`) — use these for all new public APIs.
