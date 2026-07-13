# Skill: Add an HFT Execution Path (C++)

Add ultra-low-latency execution / matching logic to the C++ HFT Native Loop (`cpp/`). **All**
sub-microsecond and HFT-venue code is C++ — never add it to Rust or Go.

1. **Data-Oriented Design**: struct-of-arrays layouts, contiguous pools, index handles (not
   pointers), cache-line alignment on hot structs. No dynamic allocation in steady state
   (pre-allocated arenas).
2. **Hot path discipline**: no syscalls, no logging, no locks on the tick-to-order path. Busy-spin
   thread pinned to an isolated core.
3. **Shared-memory only**: publish metrics/telemetry into the `shm_open`/`mmap` block via seqlock
   or double-buffer so the Rust reader never tears. **No sockets/WebSockets/loopback** for C++ IPC.
4. **Layout parity**: the POD structs mirror the Protobuf schema (`proto/`); document any mapping.
5. **Tests + bench**: Catch2/GoogleTest unit tests (`just hft::test`) and Google Benchmark
   micro-bench gating p50/p99 (`just bench::hft`).
6. **Docs**: tick `moon/roadmaps/hft_cpp.md`; record latency numbers in the PR + CHANGELOG.
