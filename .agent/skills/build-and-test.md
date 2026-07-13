# Skill: Build & Test

Standard verification sequence — run before declaring a task done.

```bash
just lint          # cargo clippy + ruff + eslint + go vet
just test-all      # every tier: rust / python / typescript / crypto (go) / hft (cpp)
```

Per-tier (faster iteration):

```bash
just build::rust      just test::rust
just crypto::build    just crypto::test     # go test -race
just hft::build       just hft::test        # cmake + ctest
just build::typescript just test::typescript
```

Notes:
- The Go and C++ recipes no-op gracefully until those modules are initialised, so `test-all`
  won't hard-fail on a fresh checkout.
- Hot-path (C++) changes must include benchmark numbers: `just bench::hft`.
- Coverage: `just coverage` (Rust + Python).
