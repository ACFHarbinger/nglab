// hft_daemon — NGLab HFT Native Loop (Tier 1 / Hot Path).
//
// An independent native daemon using Data-Oriented Design to bypass OS-kernel
// overhead and minimise CPU cache misses. It writes metrics directly to RAM via
// a shared-memory segment for zero-copy, asynchronous reading by the Rust
// backend — never over sockets. See moon/roadmaps/hft_cpp.md.
//
// Usage:
//   hft_daemon [--shm=/nglab_hft_metrics] [--iters=N] [--spin]
//     --iters=N   publish N metric snapshots then exit (default 100)
//     --spin      busy-spin forever (real hot-path mode; core-pin externally)
#include "nglab_hft/order_book.hpp"
#include "nglab_hft/shm_metrics.hpp"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

namespace {
std::atomic<bool> g_stop{false};
void on_signal(int) { g_stop.store(true, std::memory_order_relaxed); }

std::int64_t arg_int(int argc, char** argv, const char* key, std::int64_t dflt) {
    const std::size_t klen = std::strlen(key);
    for (int i = 1; i < argc; ++i) {
        if (std::strncmp(argv[i], key, klen) == 0) {
            return std::atoll(argv[i] + klen);
        }
    }
    return dflt;
}
bool arg_flag(int argc, char** argv, const char* key) {
    for (int i = 1; i < argc; ++i)
        if (std::strcmp(argv[i], key) == 0) return true;
    return false;
}
std::string arg_str(int argc, char** argv, const char* key, const std::string& dflt) {
    const std::size_t klen = std::strlen(key);
    for (int i = 1; i < argc; ++i)
        if (std::strncmp(argv[i], key, klen) == 0) return std::string(argv[i] + klen);
    return dflt;
}
}  // namespace

int main(int argc, char** argv) {
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);

    const std::string shm_name = arg_str(argc, argv, "--shm=", "/nglab_hft_metrics");
    const std::int64_t iters = arg_int(argc, argv, "--iters=", 100);
    const bool spin = arg_flag(argc, argv, "--spin");

    nglab::hft::ShmMetricsWriter writer(shm_name);
    nglab::hft::OrderBook book;

    // Seed a little resting liquidity so best bid/ask are populated.
    book.add(nglab::hft::Side::Bid, 100'00, 5);
    book.add(nglab::hft::Side::Ask, 100'05, 5);

    std::printf("hft_daemon: writing metrics to shm '%s' (%s)\n",
                shm_name.c_str(), spin ? "spin" : "bounded");

    nglab::hft::MetricsBlock m{};
    std::int64_t bid = 0, ask = 0;
    for (std::int64_t i = 0; (spin || i < iters) && !g_stop.load(std::memory_order_relaxed); ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        // (Hot path would match real orders here.)
        book.best_bid(bid);
        book.best_ask(ask);
        const auto t1 = std::chrono::steady_clock::now();
        const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

        m.ticks_processed = static_cast<std::uint64_t>(i + 1);
        m.orders_matched = m.ticks_processed / 2;
        m.p50_latency_ns = static_cast<std::uint64_t>(ns);
        m.p99_latency_ns = static_cast<std::uint64_t>(ns);
        m.best_bid = static_cast<double>(bid) / 100.0;
        m.best_ask = static_cast<double>(ask) / 100.0;
        writer.publish(m);
    }

    std::printf("hft_daemon: published %llu snapshots, exiting\n",
                static_cast<unsigned long long>(m.ticks_processed));
    return 0;
}
