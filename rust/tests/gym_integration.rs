use nglab::simulation::gym::TradingEnv;

#[test]
fn test_gym_full_episode() {
    // 1. Setup Environment
    let mut env = TradingEnv::new(100_000.0, 0.001, 5, 100, false, None);

    // 2. Load Price Data (Simple sine wave pattern)
    let prices: Vec<f64> = (0..20).map(|i| 100.0 + (i as f64).sin() * 10.0).collect();
    env.load_prices(prices.clone());

    // 3. Reset
    let initial_obs = env.reset_rs();
    // Observation should be flattened vector of size lookback * num_features
    assert_eq!(initial_obs.len(), 5 * 6);

    // Check initial observation state
    // First 5 steps are padding/pre-fill.
    // Price at index 0 should be normalized relative to initial price.

    // 4. Run Episode
    let mut total_reward = 0.0;
    let mut done = false;
    let mut steps = 0;

    // Strategy: Buy if price is low, Sell if price is high (relative to 100)
    while !done {
        let current_price_idx = 5 + steps; // lookback + steps
        if current_price_idx >= prices.len() {
            break; // Should have terminated
        }

        let current_price = prices[current_price_idx];

        let action = if current_price < 95.0 {
            1 // Buy
        } else if current_price > 105.0 {
            2 // Sell
        } else {
            0 // Hold
        };

        // Execute Step (Pure Rust version)
        let (_obs, reward, terminated, truncated, info) = env.step_rs(action);

        total_reward += reward;
        steps += 1;

        if terminated || truncated {
            done = true;
        }

        // Invariant checks
        assert!(info.cash >= 0.0, "Cash should not be negative");
        assert!(
            info.portfolio_value > 0.0,
            "Portfolio should not be bankrupt"
        );
    }

    assert!(steps > 0, "Should have taken steps");
    assert!(steps <= 20, "Should not exceed price history length");

    // We expect some non-zero reward because we traded
    println!("Total Reward: {}", total_reward);
}

#[test]
fn test_gym_observation_integrity() {
    let mut env = TradingEnv::new(10_000.0, 0.0, 2, 10, false, None); // Lookback 2
    let prices = vec![100.0, 101.0, 102.0, 103.0, 104.0];
    env.load_prices(prices);

    env.reset_rs();

    // Step 1: Hold (Price moves 102 -> 103)
    let (obs, _, _, _, _) = env.step_rs(0);

    // Observation shape is (2, 6) -> flattened to 12
    // Row 0: Time t-1 (Price 102) -> Normalized 102/100 = 1.02
    // Row 1: Time t (Price 103)   -> Normalized 103/100 = 1.03

    // Check Price Feature (Index 0 in each row)
    // Row 0: Index 1 -> 101.0 / 100.0 = 1.01
    // Row 1: Index 2 -> 102.0 / 100.0 = 1.02
    assert!(
        (obs[0] - 1.01).abs() < 1e-6,
        "Row 0 price mismatch: {}",
        obs[0]
    );
    assert!(
        (obs[6] - 1.02).abs() < 1e-6,
        "Row 1 price mismatch: {}",
        obs[6]
    );
}
