use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use nglab::simulation::orderbook::{OrderBook, Order, OrderSide, OrderType};

fn orderbook_insert_limit_orders(c: &mut Criterion) {
    let mut group = c.benchmark_group("orderbook_insert");

    for size in [100, 500, 1000, 5000].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter(|| {
                let mut book = OrderBook::new();
                for i in 0..size {
                    let order = Order::new(
                        i as u64,
                        OrderSide::Buy,
                        OrderType::Limit,
                        100.0 + (i as f64 * 0.01),
                        10,
                    );
                    book.add_order(black_box(order));
                }
                book
            });
        });
    }
    group.finish();
}

fn orderbook_match_orders(c: &mut Criterion) {
    c.bench_function("orderbook_match_simple", |b| {
        b.iter(|| {
            let mut book = OrderBook::new();

            // Add buy orders
            for i in 0..100 {
                let order = Order::new(
                    i,
                    OrderSide::Buy,
                    OrderType::Limit,
                    100.0 - (i as f64 * 0.1),
                    10,
                );
                book.add_order(order);
            }

            // Add matching sell order
            let sell_order = Order::new(
                1000,
                OrderSide::Sell,
                OrderType::Market,
                0.0,
                50,
            );
            book.add_order(black_box(sell_order));

            book
        });
    });
}

fn orderbook_price_levels(c: &mut Criterion) {
    let mut book = OrderBook::new();

    // Populate with orders
    for i in 0..1000 {
        let side = if i % 2 == 0 { OrderSide::Buy } else { OrderSide::Sell };
        let price = if side == OrderSide::Buy {
            100.0 - (i as f64 * 0.01)
        } else {
            100.0 + (i as f64 * 0.01)
        };

        let order = Order::new(
            i as u64,
            side,
            OrderType::Limit,
            price,
            10,
        );
        book.add_order(order);
    }

    c.bench_function("orderbook_get_levels", |b| {
        b.iter(|| {
            let levels = book.get_price_levels(black_box(10));
            black_box(levels)
        });
    });
}

fn orderbook_mixed_operations(c: &mut Criterion) {
    c.bench_function("orderbook_mixed_ops", |b| {
        b.iter(|| {
            let mut book = OrderBook::new();

            // Mixed operations: insert, match, cancel
            for i in 0..100 {
                // Insert limit order
                let order = Order::new(
                    i,
                    OrderSide::Buy,
                    OrderType::Limit,
                    100.0 + (i as f64 * 0.01),
                    10,
                );
                book.add_order(order);

                // Every 10th order, insert a market order to trigger matching
                if i % 10 == 0 {
                    let market_order = Order::new(
                        i + 10000,
                        OrderSide::Sell,
                        OrderType::Market,
                        0.0,
                        5,
                    );
                    book.add_order(market_order);
                }

                // Every 20th order, cancel an order
                if i % 20 == 0 && i > 0 {
                    book.cancel_order(i - 1);
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
