# HFT Native Loop (C++) — Tier 1 / Hot Path

The C++ "HFT Native Loop" is an independent native daemon for ultra-low-latency execution and raw
order-book matching (migrated out of Rust). It uses **Data-Oriented Design (DOD)** to bypass
OS-kernel overhead and minimise CPU cache misses, and writes metrics **directly to RAM via shared
memory** for zero-copy, asynchronous reading by the Rust backend.

> **Language mandate:** all future ultra-low-latency / HFT-venue work is **C++**. IPC to Rust is
> shared memory only — never sockets. See [`moon/roadmaps/hft_cpp.md`](../moon/roadmaps/hft_cpp.md).

## Layout

```
cpp/
├── CMakeLists.txt              # module env file (C++20, -O3 -march=native)
├── include/
│   ├── order_book.hpp          # DOD (struct-of-arrays) order book
│   └── shm_metrics.hpp         # shm_open/mmap seqlock metrics bridge
├── src/{order_book,shm_metrics,main}.cpp
└── tests/test_hft.cpp          # self-contained (ctest, no external framework)
```

## Build, run, test

```bash
just hft::build                 # cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release && cmake --build
just hft::run --iters=1000      # publishes metric snapshots into /nglab_hft_metrics
just hft::test                  # cmake (Debug) + ctest
```

The repo-root `CMakeLists.txt` also configures this tree (`cmake -S . -B build`).

## IPC contract (shared memory)

- The daemon is the **single writer** of a POSIX shm segment (`shm_open` + `mmap`), default name
  `/nglab_hft_metrics`.
- `MetricsBlock` is cache-line aligned and published behind a **seqlock** (`seq` even = stable,
  odd = writing) so the Rust reader never tears — zero-copy, lock-free.
- The block layout mirrors a fixed Protobuf-defined struct
  ([schema](../moon/roadmaps/schema_protobuf.md)); keep the two in lockstep.
