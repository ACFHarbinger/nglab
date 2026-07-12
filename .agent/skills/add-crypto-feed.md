# Skill: Add a Crypto Exchange Feed (Go)

Add a new exchange WebSocket / JSON-RPC feed to the Go Crypto Daemon (`go/`). **All** crypto
market-data and exchange-connection code is Go — never add it to Rust or C++.

1. **Placement**: connector under `go/internal/feeds/<exchange>/`; register it with the feed
   dispatcher in `go/internal/feeds`.
2. **Concurrency**: one goroutine per stream; bounded channel to the fan-in dispatcher; jittered
   reconnect/backoff; per-exchange rate limiter. Never block the tick path — drop-oldest with a
   counter on overflow.
3. **Decode → Protobuf**: normalize raw messages into the universal `Tick`/`Order` Protobuf types
   (`go/internal/pb`); emit length-prefixed frames over the loopback bridge.
4. **Config**: exchange endpoint/keys from env or config, never hard-coded secrets.
5. **Tests**: table-driven decode tests + framing tests; run `just crypto::test` (race detector).
6. **Docs**: tick the item in `moon/roadmaps/crypto_go.md`; update CHANGELOG when it ships.
