use nglab::simulation::orderbook::{OrderBook, Side};

#[test]
fn test_cascading_fills() {
    let mut book = OrderBook::new();

    // Reset timestamp
    book.set_timestamp(1000);

    // Seed the book with asks at increasing prices
    // Ask 1: 10 units @ 100.0
    book.submit_limit_order(100.0, 10.0, Side::Ask);
    // Ask 2: 20 units @ 101.0
    book.submit_limit_order(101.0, 20.0, Side::Ask);
    // Ask 3: 50 units @ 102.0
    book.submit_limit_order(102.0, 50.0, Side::Ask);

    assert_eq!(book.best_ask(), Some(100.0));
    assert_eq!(book.total_ask_volume(), 80.0);

    // Submit a large Market Buy order for 35 units
    // Should fill:
    // 1. 10.0 units @ 100.0 (Ask 1 cleared)
    // 2. 20.0 units @ 101.0 (Ask 2 cleared)
    // 3. 5.0 units @ 102.0 (Partial fill of Ask 3)
    let (_, trades) = book.submit_market_order(35.0, Side::Bid);

    assert_eq!(trades.len(), 3, "Should have 3 trades");

    // Verify Trade 1
    assert_eq!(trades[0].price, 100.0);
    assert_eq!(trades[0].quantity, 10.0);

    // Verify Trade 2
    assert_eq!(trades[1].price, 101.0);
    assert_eq!(trades[1].quantity, 20.0);

    // Verify Trade 3
    assert_eq!(trades[2].price, 102.0);
    assert_eq!(trades[2].quantity, 5.0);

    // Verify book state
    assert_eq!(book.best_ask(), Some(102.0));
    // Remaining volume on Ask 3: 50 - 5 = 45
    assert_eq!(book.total_ask_volume(), 45.0);
}

#[test]
fn test_price_time_priority() {
    let mut book = OrderBook::new();
    book.set_timestamp(1000);

    // Submit two limit bids at the same price
    // Bid 1: 10 units @ 100.0 (First)
    let (id1, _) = book.submit_limit_order(100.0, 10.0, Side::Bid);

    // Advance time slightly
    book.set_timestamp(1001);

    // Bid 2: 10 units @ 100.0 (Second)
    let (id2, _) = book.submit_limit_order(100.0, 10.0, Side::Bid);

    assert_eq!(book.total_bid_volume(), 20.0);

    // Submit a matching Ask for 15 units @ 100.0
    // Should fill Bid 1 completely (10 units) and Bid 2 partially (5 units)
    let (_, trades) = book.submit_limit_order(100.0, 15.0, Side::Ask);

    assert_eq!(trades.len(), 2, "Should have 2 trades");

    // Check Maker ID of first trade (should be Bid 1)
    assert_eq!(
        trades[0].maker_order_id, id1,
        "First trade should be against first bid"
    );
    assert_eq!(trades[0].quantity, 10.0);

    // Check Maker ID of second trade (should be Bid 2)
    assert_eq!(
        trades[1].maker_order_id, id2,
        "Second trade should be against second bid"
    );
    assert_eq!(trades[1].quantity, 5.0);

    // Remaining volume should be 5.0 (from Bid 2)
    assert_eq!(book.total_bid_volume(), 5.0);
}

#[test]
fn test_market_depth_view() {
    let mut book = OrderBook::new();

    // Add asks
    book.submit_limit_order(105.0, 100.0, Side::Ask);
    book.submit_limit_order(104.0, 50.0, Side::Ask);
    book.submit_limit_order(103.0, 10.0, Side::Ask);

    // Add bids
    book.submit_limit_order(100.0, 20.0, Side::Bid);
    book.submit_limit_order(99.0, 30.0, Side::Bid);
    book.submit_limit_order(98.0, 40.0, Side::Bid);

    // Check Ask Depth (ascending price)
    let ask_depth = book.ask_depth(5);
    // Should be ordered: 103, 104, 105
    assert_eq!(ask_depth.len(), 3);
    assert_eq!(ask_depth[0].0, 103.0);
    assert_eq!(ask_depth[0].1, 10.0);
    assert_eq!(ask_depth[1].0, 104.0);
    assert_eq!(ask_depth[1].1, 50.0);

    // Check Bid Depth (descending price)
    let bid_depth = book.bid_depth(5);
    // Should be ordered: 100, 99, 98
    assert_eq!(bid_depth.len(), 3);
    assert_eq!(bid_depth[0].0, 100.0);
    assert_eq!(bid_depth[0].1, 20.0);
    assert_eq!(bid_depth[1].0, 99.0);

    // Test depth limit
    let shallow_bid_depth = book.bid_depth(1);
    assert_eq!(shallow_bid_depth.len(), 1);
    assert_eq!(shallow_bid_depth[0].0, 100.0);
}
