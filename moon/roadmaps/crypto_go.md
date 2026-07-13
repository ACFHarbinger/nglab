# Roadmap — Crypto Daemon (Go, Tier 2 / Warm Path)

> **Language mandate:** ALL crypto market-data, exchange-connection, and concurrent crypto
> trading logic MUST be written in **Go**. Migrated out of Rust.

The Go "Crypto Daemon" owns I/O-bound work and massive concurrency — thousands of exchange
WebSocket feeds and JSON-RPC node connections — and streams fast market ticks and account/balance
state to the Rust hub over a local loopback bridge. Implementation lives in [`go/`](../../go/).

## §1 — Migration from Rust (Priority 0)

- [ ] Inventory existing Rust crypto logic: exchange WS clients, JSON-RPC node connectors,
      order routing, balance tracking.
- [ ] Port exchange WebSocket feed handling to Go goroutines (one goroutine per feed; back-pressured
      channels; automatic reconnect with jittered backoff).
- [ ] Port JSON-RPC node connections (EVM/chain RPC used for crypto balances/nonces).
- [ ] Port concurrent crypto trading logic (order placement/cancel, rate-limit aware).
- [ ] Remove the migrated crypto paths from the Rust crate once parity is verified.

## §2 — Loopback IPC bridge (Priority 0)

- [ ] Headless binary (`go/cmd/crypto-daemon`) accepts a dynamic port from Rust at startup:
      `crypto-daemon --port=54321`. The port is chosen by the Rust backend to avoid permission
      errors from binding privileged/fixed ports.
- [ ] Bind a loopback server on `127.0.0.1:<port>` only (never `0.0.0.0`).
- [ ] Push fast market ticks and account/balance state frames to the Rust backend over the
      loopback connection (length-prefixed Protobuf frames — see
      [schema_protobuf.md](schema_protobuf.md)).
- [ ] Heartbeat + graceful shutdown on SIGTERM so the Rust lifecycle manager can restart cleanly.

## §3 — Concurrency & resilience

- [ ] Fan-in market ticks from N feeds through a bounded dispatcher; drop-oldest on overflow with
      a dropped-frame counter (never block the hot path).
- [ ] Per-exchange rate limiters and circuit breakers.
- [ ] Structured logging (zap/slog) + Prometheus metrics endpoint (feeds up, reconnects, lag).

## §4 — Tooling & tests

- [ ] `go.mod` module `github.com/ACFHarbinger/nglab/go` (see [env files](../../go/go.mod)).
- [ ] Table-driven unit tests for feed decoding and the loopback framing; race detector in CI
      (`go test -race`).
- [ ] `just crypto::build` / `just crypto::test` / `just crypto::run` recipes.

## §5 — Future (Go-only, enforced)

- [ ] Additional CEX/DEX connectors, mempool streaming, cross-exchange arbitrage scanning — all Go.
