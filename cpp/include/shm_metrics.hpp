// Shared-memory metrics bridge for the NGLab HFT Native Loop (Tier 1).
//
// IPC rule (moon/roadmaps/hft_cpp.md §2): the C++ engine is the SINGLE WRITER of
// a POSIX shared-memory segment (shm_open + mmap); the Rust backend is a
// zero-copy, asynchronous READER. No sockets/WebSockets are used for C++ IPC.
//
// Tearing is prevented with a seqlock: the writer bumps `seq` to odd, writes the
// payload, then bumps `seq` to even. A reader samples `seq`, copies, and re-reads
// `seq`; if it changed or was odd, it retries.
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>

namespace nglab::hft {

// Cache-line-aligned telemetry block written into shared memory. POD layout so
// it mirrors a fixed Protobuf-defined struct on the Rust side.
struct alignas(64) MetricsBlock {
    std::atomic<uint64_t> seq;   // seqlock counter (even = stable, odd = writing)
    uint64_t ticks_processed;
    uint64_t orders_matched;
    uint64_t p50_latency_ns;
    uint64_t p99_latency_ns;
    int64_t  last_write_tsc;     // TSC timestamp of the last publish
    double   best_bid;
    double   best_ask;
};

// Owns a shm segment and publishes MetricsBlock snapshots via the seqlock.
class ShmMetricsWriter {
public:
    // Creates/opens the named segment (e.g. "/nglab_hft_metrics") and maps it.
    explicit ShmMetricsWriter(std::string name);
    ~ShmMetricsWriter();

    ShmMetricsWriter(const ShmMetricsWriter&) = delete;
    ShmMetricsWriter& operator=(const ShmMetricsWriter&) = delete;

    // Atomically publish a snapshot (seqlock write). Reader-side never tears.
    void publish(const MetricsBlock& snapshot) noexcept;

    const std::string& name() const noexcept { return name_; }

private:
    std::string   name_;
    int           fd_{-1};
    void*         addr_{nullptr};
    std::size_t   size_{0};
    MetricsBlock* block_{nullptr};
};

// Reader helper (used by tests here; the production reader is in Rust).
// Returns true and fills `out` with a stable (non-torn) snapshot.
bool read_stable(const MetricsBlock& block, MetricsBlock& out) noexcept;

}  // namespace nglab::hft
