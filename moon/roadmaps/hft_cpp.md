# Roadmap — HFT Native Loop (C++, Tier 1 / Hot Path)

> **Language mandate:** ALL ultra-low-latency execution and traditional HFT-venue logic MUST be
> written in **C++**. Migrated out of Rust.

The C++ "HFT Native Loop" is an independent native daemon using **Data-Oriented Design (DOD)** to
bypass OS-kernel overhead and eliminate CPU cache misses. It never uses WebSockets or network
loopbacks for IPC — it writes metrics directly to RAM via shared memory for zero-copy,
asynchronous reading by the Rust backend. Implementation lives in [`cpp/`](../../cpp/).

## §1 — Migration from Rust (Priority 0)

- [ ] Inventory existing Rust HFT logic: sub-microsecond execution loops, raw order-book matching,
      Tier-1 execution paths.
- [ ] Port the matching engine to a DOD layout: struct-of-arrays price levels, contiguous order
      pools, index handles instead of pointers, cache-line-aligned hot structures.
- [ ] Port the execution loop as a busy-spin thread pinned to an isolated core (no syscalls on the
      hot path; pre-allocated arenas; no dynamic allocation in steady state).
- [ ] Remove the migrated HFT paths from the Rust crate once parity + latency are verified.

## §2 — Shared-memory IPC bridge (Priority 0)

- [ ] Create a shared-memory segment with `shm_open` + `mmap` (or a `mmap`ed file); the C++ engine
      is the single writer, the Rust backend the reader — **zero-copy, lock-free**.
- [ ] Define a cache-line-aligned metrics/telemetry block (seqlock or double-buffer so readers never
      tear); layout matches the Protobuf-defined structs where applicable
      ([schema_protobuf.md](schema_protobuf.md)).
- [ ] **Do NOT** add sockets/WebSockets/loopback for C++ IPC — RAM only.
- [ ] Publish the segment name/size so the Rust lifecycle manager can attach.

## §3 — Latency engineering

- [ ] Core pinning + `isolcpus`/`nohz_full` guidance; huge pages for the arena.
- [ ] Branch-prediction-friendly matching; prefetch next price level.
- [ ] Nanosecond timestamping (TSC) and a latency histogram written into the shm metrics block.

## §4 — Build & tests

- [ ] `cpp/CMakeLists.txt` (C++20, `-O3 -march=native`, IPO/LTO) — see [env files](../../cpp/CMakeLists.txt).
- [ ] Unit tests (Catch2/GoogleTest) for the matcher; micro-benchmarks (Google Benchmark) gating
      p50/p99 latency regressions.
- [ ] `just hft::build` / `just hft::test` / `just hft::run` recipes.

## §5 — Future (C++-only, enforced)

- [ ] FIX/binary-venue gateways, kernel-bypass NICs (DPDK/io_uring), FPGA offload hooks — all C++.
