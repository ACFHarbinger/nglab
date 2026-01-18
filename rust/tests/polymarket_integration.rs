use nglab::simulation::polymarket::PolymarketArena;

#[test]
fn test_polymarket_full_lifecycle() {
    let mut arena = PolymarketArena::new(10_000.0, 0.0); // 0% fee for simpler math

    // 1. Load Markets (JSON)
    let markets_json = r#"[
        {
            "id": 123,
            "_filename": "market_123.json",
            "title": "Will Bitcoin hit $100k?",
            "category": "Crypto",
            "options": ["Yes", "No"],
            "outcome": null
        }
    ]"#;
    arena
        .load_markets(markets_json)
        .expect("Failed to load markets");
    assert_eq!(arena.num_markets(), 1);

    // 2. Load Price History (CSV)
    // Format: "index","timestamp","price"
    let csv_data = "\
idx,timestamp,price
0,\"1000\",\"0.40\"
1,\"1001\",\"0.50\"
2,\"1002\",\"0.60\"
";
    arena
        .load_price_history("123", csv_data)
        .expect("Failed to load history");

    assert_eq!(arena.total_ticks(), 3);
    assert_eq!(arena.get_price("123"), Some(0.40));

    // 3. Trade Sequence
    // Step 0: Price 0.40
    // Buy 1000 'Yes' tokens. Cost = 1000 * 0.40 = 400.0
    let cost = arena.buy_yes("123", 1000.0).expect("Buy failed");
    assert_eq!(cost, 400.0);
    assert_eq!(arena.collateral(), 9600.0);

    let (yes, no) = arena.get_position("123");
    assert_eq!(yes, 1000.0);
    assert_eq!(no, 0.0);

    // Advance to Step 1: Price 0.50
    assert!(arena.advance());
    assert_eq!(arena.get_price("123"), Some(0.50));

    // Unrealized PnL:
    // Position 1000 Yes. Value = 1000 * 0.50 = 500.0
    // Cost basis = 400.0
    // Unrealized PnL = +100.0
    // Account Value = Collateral (9600) + Position Value (500) = 10100.0
    assert_eq!(arena.account_value(), 10100.0);

    // Advance to Step 2: Price 0.60
    assert!(arena.advance());
    assert_eq!(arena.get_price("123"), Some(0.60));

    // Sell 500 'Yes' tokens.
    // Proceeds = 500 * 0.60 = 300.0
    let proceeds = arena.sell_yes("123", 500.0).expect("Sell failed");
    assert_eq!(proceeds, 300.0);

    // Collateral = 9600 + 300 = 9900.0
    assert_eq!(arena.collateral(), 9900.0);

    // Remaining Position: 500 Yes.
    // Current Value: 500 * 0.60 = 300.0
    // Total Account Value = 9900 + 300 = 10200.0
    assert_eq!(arena.account_value(), 10200.0);

    // Cannot advance further
    assert!(!arena.advance());
}

#[test]
fn test_insufficient_funds() {
    let mut arena = PolymarketArena::new(100.0, 0.0);

    // Setup dummy price
    let markets_json = r#"[{"id":1, "_filename":"1.json", "title":"T", "category":"C", "options":[], "outcome":null}]"#;
    arena.load_markets(markets_json).unwrap();

    let csv_data = "i,t,p\n0,\"0\",\"0.5\"";
    arena.load_price_history("1", csv_data).unwrap();

    // Try to buy $200 worth (requires 400 units at 0.5 price)
    let result = arena.buy_yes("1", 400.0);
    assert!(result.is_err(), "Should fail with insufficient funds");
}
