# Crypto Daemon (Go) — Tier 2 / Warm Path

The Go "Crypto Daemon" owns all crypto market-data, exchange connections, and concurrent trading
logic (migrated out of Rust). It handles I/O-bound work and massive concurrency — thousands of
exchange WebSocket feeds and JSON-RPC node connections — and streams fast market ticks and
account/balance state to the Rust hub over a **local loopback bridge**.

> **Language mandate:** all future crypto/exchange/concurrent-feed work is **Go**. See
> [`moon/roadmaps/crypto_go.md`](../moon/roadmaps/crypto_go.md).

## Layout

```
go/
├── go.mod
├── cmd/crypto-daemon/      # headless binary (accepts --port from Rust)
└── internal/
    ├── loopback/           # 127.0.0.1:<port> bridge, length-prefixed frames
    ├── feeds/              # exchange WS feed fan-in dispatcher
    ├── rpc/                # JSON-RPC node connections
    └── pb/                 # generated Protobuf types (schema in proto/)
```

## Run

The Rust backend supplies a dynamic loopback port at startup (avoids privileged-bind permission
errors):

```bash
just crypto::run 54321        # or: go run ./cmd/crypto-daemon --port=54321
```

The server binds `127.0.0.1:<port>` only — never `0.0.0.0`.

## Build & test

```bash
just crypto::build            # go build -o build/crypto-daemon ./cmd/crypto-daemon
just crypto::test             # go test -race ./...
```

## IPC contract

- Transport: loopback TCP; frames are 4-byte big-endian length + payload.
- Payload: serialized Protobuf `Tick` / balance messages (schema in
  [`proto/`](../moon/roadmaps/schema_protobuf.md)).
- Never use shared memory here — that transport is reserved for the C++ HFT loop.
