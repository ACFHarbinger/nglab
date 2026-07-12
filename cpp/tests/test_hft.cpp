// Self-contained tests for the HFT core (no external framework, so ctest runs
// offline). Returns non-zero on the first failure.
#include "order_book.hpp"
#include "shm_metrics.hpp"

#include <cstdio>
#include <cstdlib>

namespace {
int g_failures = 0;

void check(bool cond, const char* msg) {
    if (!cond) {
        std::fprintf(stderr, "FAIL: %s\n", msg);
        ++g_failures;
    }
}
}  // namespace

using namespace nglab::hft;

static void test_order_book_best_and_match() {
    OrderBook book;
    std::int64_t px = 0;

    check(!book.best_bid(px), "empty book has no best bid");

    book.add(Side::Bid, 100'00, 5);
    book.add(Side::Bid, 100'02, 3);
    book.add(Side::Ask, 100'05, 4);

    check(book.best_bid(px) && px == 100'02, "best bid is the highest bid");
    check(book.best_ask(px) && px == 100'05, "best ask is the lowest ask");

    // Aggressive ask for 6 @ 100.00 crosses both bids (100'02 then 100'00): fills 6.
    const std::int64_t filled = book.match(Side::Ask, 100'00, 6);
    check(filled == 6, "aggressive ask fills against resting bids");
}

static void test_shm_metrics_seqlock_round_trip() {
    ShmMetricsWriter writer("/nglab_hft_test_metrics");

    MetricsBlock snap{};
    snap.ticks_processed = 42;
    snap.orders_matched = 21;
    snap.best_bid = 100.02;
    snap.best_ask = 100.05;
    writer.publish(snap);

    // The writer's block lives in shm; re-open a reader view via a fresh mapping
    // is Rust's job. Here we validate the seqlock read helper against a local
    // block that mirrors a published one.
    MetricsBlock block{};
    block.seq.store(0, std::memory_order_relaxed);
    // simulate a completed publish (even seq) with data
    block.ticks_processed = 42;
    block.best_bid = 100.02;
    block.seq.store(2, std::memory_order_release);

    MetricsBlock out{};
    check(read_stable(block, out), "read_stable succeeds on a stable block");
    check(out.ticks_processed == 42, "read_stable copies ticks_processed");
    check(out.best_bid == 100.02, "read_stable copies best_bid");
}

int main() {
    test_order_book_best_and_match();
    test_shm_metrics_seqlock_round_trip();

    if (g_failures == 0) {
        std::printf("all HFT tests passed\n");
        return EXIT_SUCCESS;
    }
    std::fprintf(stderr, "%d HFT test(s) failed\n", g_failures);
    return EXIT_FAILURE;
}
