use nglab::simulation::multi_asset::MultiAssetEnv;
use nglab::simulation::orderbook::Side;
use nglab::simulation::spreads::{Leg, SpreadOrder, SpreadType};

#[test]
fn test_vertical_spread_execution() {
    let assets = vec!["BTC".to_string(), "ETH".to_string()];
    let mut env = MultiAssetEnv::new(assets, 100_000.0, 0.0, 1, 100, Some(42));

    // 1. Create Liquidity
    // BTC Price ~ 50,000. Long leg needs to buy at Ask.
    // ETH Price ~ 3,000. Short leg needs to sell at Bid.

    // Seed Orderbooks
    env.seed_orderbook("BTC", 50000.0).unwrap();
    env.seed_orderbook("ETH", 3000.0).unwrap();

    // 2. Create Spread Order
    // Long 1 BTC, Short 10 ETH.
    // Net Cost Limit: Pay max 20,000.
    // Cost = (1 * 50025 (Ask)) - (10 * 2998.5 (Bid))
    // Ask = 50000 + 25 = 50025
    // Bid = 3000 - 1.5 = 2998.5
    // Cost = 50025 - 29985 = 20040.
    // If Limit is 20000, it SHOULD NOT EXECUTE.

    let spread = SpreadOrder {
        spread_type: SpreadType::Custom,
        legs: vec![
            Leg {
                asset: "BTC".to_string(),
                side: Side::Bid,
                ratio: 1.0,
            },
            Leg {
                asset: "ETH".to_string(),
                side: Side::Ask,
                ratio: 10.0,
            },
        ],
        limit_price: 20000.0, // Willing to pay 20k net
        quantity: 1.0,
    };

    env.submit_spread_order(spread.clone());

    // Step environment to trigger processing
    env.step_native(vec![0, 0]).unwrap(); // Action Hold

    // Check: Should NOT have executed
    assert_eq!(*env.positions.get("BTC").unwrap(), 0.0);

    // 3. Relax Limit
    let spread_executable = SpreadOrder {
        limit_price: 21000.0, // Willing to pay 21k
        ..spread
    };
    env.submit_spread_order(spread_executable);

    env.step_native(vec![0, 0]).unwrap();

    // Check: Should HAVE executed
    assert_eq!(*env.positions.get("BTC").unwrap(), 1.0);
    assert_eq!(*env.positions.get("ETH").unwrap(), -10.0);
}
