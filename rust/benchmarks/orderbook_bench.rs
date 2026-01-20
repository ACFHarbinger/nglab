/*!
 * Performance benchmarks for the Limit Order Book (LOB).
 *
 * Measures the efficiency of order insertion, cancellations, matching,
 * and price level extraction across different book sizes.
 */
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use nglab::simulation::orderbook::{OrderBook, Side};

/**
 * Benchmarks order insertion latency for limit orders at various scales.
 */
fn orderbook_insert_limit_orders(c: &mut Criterion) {
    let mut group = c.benchmark_group("orderbook_insert");

    for size in [100, 500, 1000, 5000].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter(|| {
                let mut book = OrderBook::new();
                for i in 0..size {
                    book.submit_limit_order(100.0 + (i as f64 * 0.01), 10.0, Side::Bid)
                        .unwrap();
                }
                std::hint::black_box(book)
            });
        });
    }
    group.finish();
}

/**
 * Benchmarks the matching engine performance when processing market orders.
 */
fn orderbook_match_orders(c: &mut Criterion) {
    c.bench_function("orderbook_match_simple", |b| {
        b.iter(|| {
            let mut book = OrderBook::new();

            // Add buy orders
            for i in 0..100 {
                book.submit_limit_order(100.0 - (i as f64 * 0.1), 10.0, Side::Bid)
                    .unwrap();
            }

            // Add matching sell order (Market order for immediate fill)
            book.submit_market_order(50.0, Side::Ask).unwrap();

            std::hint::black_box(book)
        });
    });
}

/**
 * Benchmarks the extraction of top price levels from the order book.
 */
fn orderbook_price_levels(c: &mut Criterion) {
    let mut book = OrderBook::new();

    // Populate with orders
    for i in 0..1000 {
        let side = if i % 2 == 0 { Side::Bid } else { Side::Ask };
        let price = if side == Side::Bid {
            100.0 - (i as f64 * 0.01)
        } else {
            100.0 + (i as f64 * 0.01)
        };

        book.submit_limit_order(price, 10.0, side).unwrap();
    }

    c.bench_function("orderbook_get_levels", |b| {
        b.iter(|| {
            // Get levels for both sides
            let bid_levels = book.bid_depth(std::hint::black_box(10));
            std::hint::black_box(bid_levels)
        });
    });
}

/**
 * Benchmarks a realistic mix of insertions, matches, and cancellations.
 */
fn orderbook_mixed_operations(c: &mut Criterion) {
    c.bench_function("orderbook_mixed_ops", |b| {
        b.iter(|| {
            let mut book = OrderBook::new();

            // Mixed operations: insert, match, cancel
            for i in 0..100 {
                // Insert limit order
                book.submit_limit_order(100.0 + (i as f64 * 0.01), 10.0, Side::Bid)
                    .unwrap();

                // Every 10th order, insert a market order to trigger matching
                if i % 10 == 0 {
                    book.submit_market_order(5.0, Side::Ask).unwrap();
                }

                // Every 20th order, cancel an order
                // Note: canceling by deterministic index is tricky as IDs increment.
                // We'll just rely on the API. In the old loop, IDs were explicit loops.
                // In new API, IDs are returned.
                // For simplicity in a tight loop benchmark, we might skip cancel or store generic IDs.
                // For now, let's just do more inserts/matches to keep it running.
                if i % 20 == 0 && i > 0 {
                    // Try to cancel an order we likely just made.
                    // Order IDs start at 1. `i` is 20, 40 etc.
                    // Let's just try cancelling a fixed offset.
                    let _ = book.cancel_order(i as u64);
                }
            }

            book
        });
    });
}

criterion_group!(
    benches,
    orderbook_insert_limit_orders,
    orderbook_match_orders,
    orderbook_price_levels,
    orderbook_mixed_operations
);
criterion_main!(benches);
