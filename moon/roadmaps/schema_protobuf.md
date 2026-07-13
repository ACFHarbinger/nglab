# Roadmap — Universal Schema (Protobuf)

All cross-boundary data structures — `Order`, `Position`, `Tick`, and telemetry frames — are
defined once in **Protocol Buffers** and code-generated for **TypeScript, Rust, Go, and C++** to
guarantee IPC boundary stability across the polyglot tiers.

## §1 — Schema definition (Priority 0)

- [ ] Create `proto/` with `order.proto`, `position.proto`, `tick.proto`, `telemetry.proto`.
- [ ] Version the package (e.g. `nglab.v1`); additive-only changes within a major version.
- [ ] Model the C++ shm metrics block and the Go loopback frames on these messages.

## §2 — Code generation (all four languages)

- [ ] Rust: `prost`/`tonic-build` in the Rust crate build.
- [ ] Go: `protoc-gen-go` into `go/internal/pb`.
- [ ] C++: `protoc` C++ output into `cpp/generated/` (or lightweight hand-mapped POD structs for
      the zero-copy shm hot path, kept in lockstep with the `.proto`).
- [ ] TypeScript: `ts-proto` into `typescript/src/pb`.
- [ ] A single `just schema::gen` recipe regenerates all four; CI fails if generated code drifts.

## §3 — Boundary contracts

- [ ] Go ↔ Rust loopback: length-prefixed serialized Protobuf frames.
- [ ] C++ ↔ Rust shm: fixed-layout POD mirror of the Protobuf structs (documented mapping;
      seqlock/double-buffered).
