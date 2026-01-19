/*!
 * Benchmarks for the `TradingEnv` simulation performance.
 *
 * Focuses on the latency of individual environment steps and full
 * episode simulations, accounting for different lookback window sizes.
 */
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use nglab::simulation::gym::TradingEnv;

/**
 * Benchmarks the performance of `env.step()` under varying lookback windows.
 */
fn env_step_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("trading_env_step");

    for lookback in [10, 50, 100, 200].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(lookback),
            lookback,
            |b, &lookback| {
                b.iter(|| {
                    // Constructor: initial_capital, transaction_cost, lookback, max_steps, logging
                    let mut env = TradingEnv::new(10000.0, 0.001, lookback, 1000, false);
                    env.reset_rs();

                    // Run 100 steps
                    for _ in 0..100 {
                        env.step_rs(std::hint::black_box(1));
                    }
                });
            },
        );
    }
    group.finish();
}

/**
 * Benchmarks the environment reset operation.
 */
fn env_reset_performance(c: &mut Criterion) {
    c.bench_function("trading_env_reset", |b| {
        let mut env = TradingEnv::new(10000.0, 0.001, 50, 1000, false);

        b.iter(|| {
            env.reset_rs();
        });
    });
}

/**
 * Benchmarks the time taken for a full 1000-step simulation episode.
 */
fn env_episode_performance(c: &mut Criterion) {
    c.bench_function("trading_env_full_episode", |b| {
        b.iter(|| {
            let mut env = TradingEnv::new(10000.0, 0.001, 50, 1000, false);
            // Need some dummy data or it terminates instantly?
            // reset_rs populates prices via generate_observation_data but uses self.prices which are empty by default?
            // Ah, env.load_prices needed for meaningful step execution.
            // Let's create dummy prices.
            let prices: Vec<f64> = (0..1100).map(|i| 100.0 + (i as f64).sin()).collect();
            env.load_prices(prices);

            env.reset_rs();

            let mut step_count = 0;
            let max_steps = 1000;

            while step_count < max_steps {
                let (_, _, terminated, truncated, _) = env.step_rs(std::hint::black_box(1));
                step_count += 1;

                if terminated || truncated {
                    break;
                }
            }

            std::hint::black_box(step_count)
        });
    });
}

/**
 * Benchmarks the overhead of reward calculation and state updates.
 */
fn env_reward_calculation(c: &mut Criterion) {
    c.bench_function("trading_env_reward_calc", |b| {
        let mut env = TradingEnv::new(10000.0, 0.001, 50, 1000, false);
        let prices: Vec<f64> = (0..200).map(|i| 100.0 + (i as f64).sin()).collect();
        env.load_prices(prices);

        env.reset_rs();

        b.iter(|| {
            // Execute a buy action
            env.step_rs(std::hint::black_box(1));
            // Execute a sell action
            env.step_rs(std::hint::black_box(2));
        });
    });
}

criterion_group!(
    benches,
    env_step_performance,
    env_reset_performance,
    env_episode_performance,
    env_reward_calculation
);
criterion_main!(benches);
