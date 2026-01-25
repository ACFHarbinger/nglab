use nglab::execution::{AlgoParams, AlgoType};
use nglab::simulation::multi_asset::MultiAssetEnv;
use nglab::simulation::orderbook::Side;

#[test]
fn test_twap_integration() {
    let assets = vec!["BTC".to_string()];
    let mut env = MultiAssetEnv::new(assets, 10000.0, 0.001, 10, 100, Some(42));
    env.load_prices("BTC".to_string(), vec![100.0; 200]);
    env.reset_native(None).unwrap();

    // Submit TWAP Order
    let start = env.current_step as u64;
    let params = AlgoParams {
        quantity: 10.0,
        side: Side::Bid,
        duration_steps: Some(4),
        urgency: None,
        participation_rate: None,
    };

    env.algo_managers
        .get_mut("BTC")
        .unwrap()
        .submit(AlgoType::TWAP, params, start);

    // Initial check
    assert_eq!(*env.positions.get("BTC").unwrap(), 0.0);

    // Step 1
    env.step_native(vec![0]).unwrap();
    // Should have executed ~2.5 shares (10.0 / 4 steps)
    // plus/minus jitter.
    let pos_1 = *env.positions.get("BTC").unwrap();
    assert!(pos_1 > 0.0);
    assert!(pos_1 < 5.0);

    // Run remaining steps
    for _ in 0..5 {
        env.step_native(vec![0]).unwrap();
    }

    // Should be fully executed
    let final_pos = *env.positions.get("BTC").unwrap();
    assert!(
        (final_pos - 10.0).abs() < 1e-6,
        "Final position should be 10.0, got {}",
        final_pos
    );

    // Cash should be reduced
    assert!(env.cash < 10000.0);
}

#[test]
fn test_pov_integration() {
    let assets = vec!["ETH".to_string()];
    // Need sufficient capital for execution (Price 2000 * 50 = 100k)
    let mut env = MultiAssetEnv::new(assets, 1_000_000.0, 0.001, 10, 100, Some(42));
    env.load_prices("ETH".to_string(), vec![2000.0; 200]);
    env.reset_native(None).unwrap();

    // Seed orderbook deeply to allow market orders
    env.seed_orderbook("ETH", 2000.0).unwrap();

    // Submit POV Order
    let start = env.current_step as u64;
    let params = AlgoParams {
        quantity: 50.0,
        side: Side::Bid,
        duration_steps: None, // Runs until filled
        urgency: None,
        participation_rate: Some(0.1), // 10% of volume
    };

    env.algo_managers
        .get_mut("ETH")
        .unwrap()
        .submit(AlgoType::POV, params, start);

    // Step
    // Note: MultiAssetEnv currently mocks volume as 1000.0 in process_algo_orders
    // So distinct step should execute 1000.0 * 0.1 = 100.0 shares.
    // Wait, mock volume is 1000.0. 10% is 100.0.
    // The order size is 50.0.
    // So it should fill completely in one step if 100.0 > 50.0.

    env.step_native(vec![0]).unwrap();

    let pos = *env.positions.get("ETH").unwrap();
    assert_eq!(pos, 50.0);
}
