/*!
 * Benchmarks for the arena simulation.
 *
 * This module contains performance benchmarks for core arena operations,
 * including order book insertions, matching logic, and environment step execution.
 *
 * Run with: `cargo bench`
 */

use criterion::{black_box, criterion_group, criterion_main, Criterion};

// Note: We need to import from the crate
// For now, we'll benchmark with inline implementations

/**
 * Benchmark suite for basic order book operations using inline simple implementations.
 */
fn benchmark_orderbook_operations(c: &mut Criterion) {
    use std::collections::BTreeMap;
    use std::collections::VecDeque;

    #[derive(Clone)]
    struct SimpleOrder {
        price: i64,
        quantity: f64,
    }

    c.bench_function("orderbook_insert_1000", |b| {
        b.iter(|| {
            let mut bids: BTreeMap<i64, VecDeque<SimpleOrder>> = BTreeMap::new();
            for i in 0..1000 {
                let price = 10000 - (i % 100);
                let order = SimpleOrder {
                    price,
                    quantity: 10.0,
                };
                bids.entry(price).or_default().push_back(order);
            }
            black_box(bids)
        })
    });

    c.bench_function("orderbook_matching_100", |b| {
        // Pre-populate order book
        let mut asks: BTreeMap<i64, VecDeque<SimpleOrder>> = BTreeMap::new();
        for i in 0..100 {
            let price = 10000 + (i % 10);
            let order = SimpleOrder {
                price,
                quantity: 10.0,
            };
            asks.entry(price).or_default().push_back(order);
        }

        b.iter(|| {
            let mut asks_clone = asks.clone();
            let mut filled = 0.0;
            let mut remaining = 500.0;

            for (&_price, orders) in asks_clone.iter_mut() {
                while let Some(order) = orders.front_mut() {
                    let fill = remaining.min(order.quantity);
                    filled += fill;
                    remaining -= fill;
                    order.quantity -= fill;
                    if order.quantity <= 0.0 {
                        orders.pop_front();
                    }
                    if remaining <= 0.0 {
                        break;
                    }
                }
                if remaining <= 0.0 {
                    break;
                }
            }
            black_box(filled)
        })
    });
}

/**
 * Benchmark suite for simulation step performance.
 *
 * Measures the time taken to execute a standard trading loop with simple
 * price-action based decision making.
 */
fn benchmark_simulation_step(c: &mut Criterion) {
    c.bench_function("env_step_1000", |b| {
        // Simulate 1000 steps of a trading environment
        let prices: Vec<f64> = (0..1100).map(|i| 100.0 + (i as f64 * 0.01).sin()).collect();

        b.iter(|| {
            let mut position = 0.0;
            let mut cash = 10000.0;
            let mut total_reward = 0.0;

            for i in 30..1030 {
                let price = prices[i];
                let prev_price = prices[i - 1];
                let returns = (price - prev_price) / prev_price;

                // Simple action: buy if returns > 0
                if returns > 0.0 && cash > 100.0 {
                    let shares = 100.0 / price;
                    cash -= 100.0;
                    position += shares;
                } else if returns < 0.0 && position > 0.0 {
                    let shares = (100.0 / price).min(position);
                    cash += shares * price;
                    position -= shares;
                }

                let portfolio_value = cash + position * price;
                total_reward += returns;
                black_box(portfolio_value);
            }
            black_box(total_reward)
        })
    });
}

criterion_group!(
    benches,
    benchmark_orderbook_operations,
    benchmark_simulation_step,
);
criterion_main!(benches);
