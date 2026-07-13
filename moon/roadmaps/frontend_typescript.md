# Roadmap — Control Panel (TypeScript, UI)

The TypeScript/React/Tauri frontend stays **thin**: a consumer of data streams and an issuer of
execution triggers. No business logic. Implementation lives in
[`typescript/`](../../typescript/).

## §1 — Boundary (enforced)

- TS strictly consumes streams from Rust (`#[tauri::command]` / events) and issues execution
  triggers. Matching, risk, and strategy logic live in the backend tiers.
- No direct exchange connections from the UI.

## §2 — Streams & triggers

- [ ] Subscribe to unified market-tick / position / order streams surfaced by the Rust hub
      (originating from the Go loopback and C++ shm bridges).
- [ ] Render order book, positions, and latency/telemetry from the C++ metrics block.
- [ ] Issue execution triggers as typed commands (Protobuf-generated TS types —
      [schema_protobuf.md](schema_protobuf.md)).

## §3 — Type safety

- [ ] Generate TS types from the universal Protobuf schema; keep them in sync in CI.
