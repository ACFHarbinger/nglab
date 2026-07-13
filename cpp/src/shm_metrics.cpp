#include "shm_metrics.hpp"

#include <fcntl.h>     // shm_open, O_*
#include <sys/mman.h>  // mmap, munmap, shm_unlink
#include <unistd.h>    // ftruncate, close

#include <cstring>
#include <stdexcept>
#include <string>

namespace nglab::hft {

ShmMetricsWriter::ShmMetricsWriter(std::string name)
    : name_(std::move(name)), size_(sizeof(MetricsBlock)) {
    // Single-writer create/truncate of a POSIX shared-memory object.
    fd_ = ::shm_open(name_.c_str(), O_CREAT | O_RDWR, 0600);
    if (fd_ < 0) {
        throw std::runtime_error("shm_open failed for " + name_);
    }
    if (::ftruncate(fd_, static_cast<off_t>(size_)) != 0) {
        ::close(fd_);
        throw std::runtime_error("ftruncate failed for " + name_);
    }
    addr_ = ::mmap(nullptr, size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
    if (addr_ == MAP_FAILED) {
        ::close(fd_);
        throw std::runtime_error("mmap failed for " + name_);
    }
    block_ = new (addr_) MetricsBlock{};
    block_->seq.store(0, std::memory_order_release);
}

ShmMetricsWriter::~ShmMetricsWriter() {
    if (addr_ != nullptr && addr_ != MAP_FAILED) {
        ::munmap(addr_, size_);
    }
    if (fd_ >= 0) {
        ::close(fd_);
        ::shm_unlink(name_.c_str());
    }
}

void ShmMetricsWriter::publish(const MetricsBlock& snapshot) noexcept {
    const uint64_t s = block_->seq.load(std::memory_order_relaxed);
    // Enter write section: make seq odd.
    block_->seq.store(s + 1, std::memory_order_release);
    std::atomic_thread_fence(std::memory_order_release);

    block_->ticks_processed = snapshot.ticks_processed;
    block_->orders_matched  = snapshot.orders_matched;
    block_->p50_latency_ns  = snapshot.p50_latency_ns;
    block_->p99_latency_ns  = snapshot.p99_latency_ns;
    block_->last_write_tsc  = snapshot.last_write_tsc;
    block_->best_bid        = snapshot.best_bid;
    block_->best_ask        = snapshot.best_ask;

    std::atomic_thread_fence(std::memory_order_release);
    // Leave write section: make seq even again.
    block_->seq.store(s + 2, std::memory_order_release);
}

bool read_stable(const MetricsBlock& block, MetricsBlock& out) noexcept {
    for (int attempt = 0; attempt < 16; ++attempt) {
        const uint64_t s1 = block.seq.load(std::memory_order_acquire);
        if (s1 & 1ULL) continue;  // writer in progress
        std::atomic_thread_fence(std::memory_order_acquire);

        out.ticks_processed = block.ticks_processed;
        out.orders_matched  = block.orders_matched;
        out.p50_latency_ns  = block.p50_latency_ns;
        out.p99_latency_ns  = block.p99_latency_ns;
        out.last_write_tsc  = block.last_write_tsc;
        out.best_bid        = block.best_bid;
        out.best_ask        = block.best_ask;

        std::atomic_thread_fence(std::memory_order_acquire);
        const uint64_t s2 = block.seq.load(std::memory_order_acquire);
        if (s1 == s2) return true;  // stable read
    }
    return false;
}

}  // namespace nglab::hft
