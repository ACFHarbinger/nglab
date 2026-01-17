/*!
 * Benchmarks for the `TradingEnv` simulation performance.
 *
 * Focuses on the latency of individual environment steps and full
 * episode simulations, accounting for different lookback window sizes.
 */
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
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
                    let mut env = TradingEnv::new(10000.0, lookback, 0.001);
                    env.reset(Some(42));

                    // Run 100 steps
                    for _ in 0..100 {
                        env.step(black_box(1));
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
        let mut env = TradingEnv::new(10000.0, 50, 0.001);

        b.iter(|| {
            env.reset(black_box(Some(42)));
        });
    });
}

/**
 * Benchmarks the time taken for a full 1000-step simulation episode.
 */
fn env_episode_performance(c: &mut Criterion) {
    c.bench_function("trading_env_full_episode", |b| {
        b.iter(|| {
            let mut env = TradingEnv::new(10000.0, 50, 0.001);
            env.reset(Some(42));

            let mut step_count = 0;
            let max_steps = 1000;

            while step_count < max_steps {
                let (_, _, terminated, truncated, _) = env.step(black_box(1));
                step_count += 1;

                if terminated || truncated {
                    break;
                }
            }

            black_box(step_count)
        });
    });
}

/**
 * Benchmarks the overhead of reward calculation and state updates.
 */
fn env_reward_calculation(c: &mut Criterion) {
    c.bench_function("trading_env_reward_calc", |b| {
        let mut env = TradingEnv::new(10000.0, 50, 0.001);
        env.reset(Some(42));

        b.iter(|| {
            // Execute a buy action
            env.step(black_box(1));
            // Execute a sell action
            env.step(black_box(2));
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
