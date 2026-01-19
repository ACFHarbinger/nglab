/*!
 * Order Book implementation with price-priority, time-priority matching.
 *
 * This module implements a Central Limit Order Book (CLOB) that serves as the matching engine
 * for the simulation. It ensures fair execution based on standard market microstructure rules.
 *
 * # Matching Rules
 *
 * 1. **Price Priority**: Better prices execute first (Higher Bids, Lower Asks).
 * 2. **Time Priority**: At the same price, earlier orders execute first (FIFO).
 *
 * # Features
 *
 * - **Limit Orders**: Orders to buy/sell at a specific price or better.
 * - **Market Orders**: Orders to buy/sell immediately at the best available price.
 * - **Snapshotting**: Ability to capture the full state of the book (L2 Data) for observation.
 */

use indexmap::IndexMap;
#[cfg(feature = "python")]
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/**
 * Order side: Bid (buy) or Ask (sell).
 */
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Side {
    Bid,
    Ask,
}

/**
 * Supported order types.
 */
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderType {
    Limit,
    Market,
    StopLoss,   // Triggers Market Sell/Buy
    TakeProfit, // Triggers Market Sell/Buy
    StopLimit,  // Triggers Limit Order
}

/**
 * A single order entry in the book.
 */
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Order {
    pub id: u64,
    pub price: f64,
    pub quantity: f64,
    pub filled: f64,
    pub side: Side,
    pub order_type: OrderType,
    pub timestamp: u64,

    // Advanced fields
    pub trigger_price: Option<f64>,    // For Stop/TakeProfit
    pub iceberg_total: Option<f64>,    // Total quantity for Iceberg (if larger than quantity)
    pub visible_quantity: Option<f64>, // Current visible slice
    pub parent_id: Option<u64>,        // For OCO (points to linked order)
}

impl Order {
    pub fn new(
        id: u64,
        price: f64,
        quantity: f64,
        side: Side,
        order_type: OrderType,
        timestamp: u64,
    ) -> Self {
        Order {
            id,
            price,
            quantity,
            filled: 0.0,
            side,
            order_type,
            timestamp,
            trigger_price: None,
            iceberg_total: None,
            visible_quantity: None,
            parent_id: None,
        }
    }

    pub fn new_advanced(
        id: u64,
        price: f64,
        quantity: f64,
        side: Side,
        order_type: OrderType,
        timestamp: u64,
        trigger_price: Option<f64>,
        iceberg_total: Option<f64>,
        visible_quantity: Option<f64>,
        parent_id: Option<u64>,
    ) -> Self {
        Order {
            id,
            price,
            quantity,
            filled: 0.0,
            side,
            order_type,
            timestamp,
            trigger_price,
            iceberg_total,
            visible_quantity,
            parent_id,
        }
    }

    /** Remaining quantity to be filled */
    pub fn remaining(&self) -> f64 {
        self.quantity - self.filled
    }

    /** Check if order is fully filled */
    pub fn is_filled(&self) -> bool {
        self.filled >= self.quantity
    }

    /// Check if this order is an iceberg order (has hidden quantity)
    pub fn is_iceberg(&self) -> bool {
        self.iceberg_total.is_some()
    }

    /// Reload visible quantity for iceberg orders from the hidden stash
    pub fn refresh_iceberg(&mut self) {
        if let (Some(total), Some(visible_max)) = (self.iceberg_total, self.visible_quantity) {
            // Deduct the currently filled amount from the total remaining
            let remaining_total = (total - self.filled).max(0.0);

            if remaining_total <= 0.0000001 {
                // Fully exhausted
                self.iceberg_total = Some(0.0);
                self.quantity = 0.0;
                self.filled = 0.0;
            } else {
                // Reload next slice
                self.iceberg_total = Some(remaining_total);
                self.quantity = remaining_total.min(visible_max);
                self.filled = 0.0;
            }
        }
    }
}
/**
 * A price level in the book, maintaining a FIFO queue of orders.
 */
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PriceLevel {
    pub price: f64,
    pub orders: VecDeque<Order>,
    pub total_quantity: f64,
}

impl PriceLevel {
    pub fn new(price: f64) -> Self {
        PriceLevel {
            price,
            orders: VecDeque::new(),
            total_quantity: 0.0,
        }
    }

    /**
     * Add a target order to this price level.
     *
     * Updates the total quantity at this level and appends the order to the queue.
     */
    pub fn add_order(&mut self, order: Order) {
        self.total_quantity += order.remaining();
        self.orders.push_back(order);
    }

    /** Remove the front order from the price level */
    pub fn remove_front(&mut self) -> Option<Order> {
        if let Some(order) = self.orders.pop_front() {
            self.total_quantity -= order.remaining();
            Some(order)
        } else {
            None
        }
    }

    /** Check if the price level is empty */
    pub fn is_empty(&self) -> bool {
        self.orders.is_empty()
    }
}

/**
 * Record of a trade execution between maker and taker orders.
 */
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    pub maker_order_id: u64,
    pub taker_order_id: u64,
    pub price: f64,
    pub quantity: f64,
    pub side: Side,
    pub timestamp: u64,
}

/**
 * Central Limit Order Book (CLOB) simulator.
 */
#[cfg_attr(feature = "python", pyclass)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBook {
    /** Bids ordered by price (highest first) */
    bids: IndexMap<i64, PriceLevel>,
    /** Asks ordered by price (lowest first) */
    asks: IndexMap<i64, PriceLevel>,
    /** Orders waiting for a trigger price (Stop/TakeProfit) */
    stop_orders: Vec<Order>,
    /** Next order ID */
    next_order_id: u64,
    /** Current timestamp */
    timestamp: u64,
    /** Price precision multiplier */
    price_precision: f64,
    /** Last trade price for trigger checking */
    last_price: f64,
}

impl Default for OrderBook {
    fn default() -> Self {
        Self::new()
    }
}

// =========================================================================
// Python Bindings Implementation
// =========================================================================

#[cfg(feature = "python")]
#[pymethods]
impl OrderBook {
    #[new]
    pub fn new_py() -> Self {
        Self::new()
    }

    #[pyo3(name = "best_bid")]
    pub fn best_bid_py(&self) -> Option<f64> {
        self.best_bid()
    }

    #[pyo3(name = "best_ask")]
    pub fn best_ask_py(&self) -> Option<f64> {
        self.best_ask()
    }

    #[pyo3(name = "mid_price")]
    pub fn mid_price_py(&self) -> Option<f64> {
        self.mid_price()
    }

    #[pyo3(name = "spread")]
    pub fn spread_py(&self) -> Option<f64> {
        self.spread()
    }

    #[pyo3(name = "imbalance")]
    pub fn imbalance_py(&self) -> f64 {
        self.imbalance()
    }

    #[pyo3(name = "total_bid_volume")]
    pub fn total_bid_volume_py(&self) -> f64 {
        self.total_bid_volume()
    }

    #[pyo3(name = "total_ask_volume")]
    pub fn total_ask_volume_py(&self) -> f64 {
        self.total_ask_volume()
    }
}

// =========================================================================
// Pure Rust Implementation
// =========================================================================

impl OrderBook {
    pub fn new() -> Self {
        OrderBook {
            bids: IndexMap::new(),
            asks: IndexMap::new(),
            stop_orders: Vec::new(),
            next_order_id: 1,
            timestamp: 0,
            price_precision: 10000.0,
            last_price: 100.0, // Default sane start
        }
    }

    /** Submit an advanced order (Stop, OCO, Iceberg) */
    pub fn submit_advanced_order(
        &mut self,
        price: f64,
        quantity: f64,
        side: Side,
        order_type: OrderType,
        trigger_price: Option<f64>,
        iceberg_total: Option<f64>,
        parent_id: Option<u64>,
    ) -> u64 {
        let order_id = self.next_order_id;
        self.next_order_id += 1;

        let order = Order::new_advanced(
            order_id,
            price,
            quantity,
            side,
            order_type,
            self.timestamp,
            trigger_price,
            iceberg_total,
            iceberg_total.map(|t| t.min(quantity)), // Visible is min(total, slice)
            parent_id,
        );

        // Handle stop orders
        if matches!(
            order_type,
            OrderType::StopLoss | OrderType::TakeProfit | OrderType::StopLimit
        ) {
            self.stop_orders.push(order);
        } else {
            // Regular/Iceberg goes to matching immediately
            self.match_order(order);
        }

        order_id
    }

    /// Check if any stop orders are triggered by current price
    pub fn check_triggers(&mut self, current_price: f64) -> Vec<Trade> {
        let mut triggered_trades = Vec::new();
        let mut activated_orders = Vec::new();

        // Update last price
        self.last_price = current_price;

        // Extract triggered orders
        // Note: retain is O(N). For high-performance, use a specialized heap/tree.
        // For NGLab simulation scale, Vec is fine.
        let mut i = 0;
        while i < self.stop_orders.len() {
            let order = &self.stop_orders[i];
            let should_trigger = match (order.side, order.trigger_price) {
                (Side::Bid, Some(tp)) => current_price >= tp, // Stop Buy (e.g. Breakout)
                (Side::Ask, Some(tp)) => current_price <= tp, // Stop Loss Sell
                _ => false,
            };

            if should_trigger {
                let mut triggered = self.stop_orders.remove(i);
                // Convert to market/limit and activate
                triggered.order_type = match triggered.order_type {
                    OrderType::StopLimit => OrderType::Limit,
                    _ => OrderType::Market,
                };
                activated_orders.push(triggered);
            } else {
                i += 1;
            }
        }

        // Execute activated orders
        for order in activated_orders {
            triggered_trades.extend(self.match_order(order));
        }

        triggered_trades
    }

    /** Get the best bid price */
    pub fn best_bid(&self) -> Option<f64> {
        self.bids
            .keys()
            .max()
            .map(|&p| p as f64 / self.price_precision)
    }

    /** Get the best ask price */
    pub fn best_ask(&self) -> Option<f64> {
        self.asks
            .keys()
            .min()
            .map(|&p| p as f64 / self.price_precision)
    }

    /** Get the mid price */
    pub fn mid_price(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some((bid + ask) / 2.0),
            _ => None,
        }
    }

    /** Get the spread */
    pub fn spread(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some(ask - bid),
            _ => None,
        }
    }

    /** Get bid depth at top N levels */
    pub fn bid_depth(&self, levels: usize) -> Vec<(f64, f64)> {
        let mut sorted_prices: Vec<_> = self.bids.keys().copied().collect();
        sorted_prices.sort_by(|a, b| b.cmp(a)); // Descending for bids

        sorted_prices
            .into_iter()
            .take(levels)
            .filter_map(|price_key| {
                self.bids.get(&price_key).map(|level| {
                    (
                        price_key as f64 / self.price_precision,
                        level.total_quantity,
                    )
                })
            })
            .collect()
    }

    /** Get ask depth at top N levels */
    pub fn ask_depth(&self, levels: usize) -> Vec<(f64, f64)> {
        let mut sorted_prices: Vec<_> = self.asks.keys().copied().collect();
        sorted_prices.sort(); // Ascending for asks

        sorted_prices
            .into_iter()
            .take(levels)
            .filter_map(|price_key| {
                self.asks.get(&price_key).map(|level| {
                    (
                        price_key as f64 / self.price_precision,
                        level.total_quantity,
                    )
                })
            })
            .collect()
    }

    /** Total volume on bid side */
    pub fn total_bid_volume(&self) -> f64 {
        self.bids.values().map(|level| level.total_quantity).sum()
    }

    /** Total volume on ask side */
    pub fn total_ask_volume(&self) -> f64 {
        self.asks.values().map(|level| level.total_quantity).sum()
    }

    /** Order book imbalance (-1 to 1, positive means more bids) */
    pub fn imbalance(&self) -> f64 {
        let bid_vol = self.total_bid_volume();
        let ask_vol = self.total_ask_volume();
        let total = bid_vol + ask_vol;
        if total == 0.0 {
            0.0
        } else {
            (bid_vol - ask_vol) / total
        }
    }

    /** Set the current timestamp */
    pub fn set_timestamp(&mut self, timestamp: u64) {
        self.timestamp = timestamp;
    }
}

impl OrderBook {
    /** Convert price to integer key for indexing */
    fn price_to_key(&self, price: f64) -> i64 {
        (price * self.price_precision).round() as i64
    }

    /** Submit a limit order and return any trades that result */
    pub fn submit_limit_order(
        &mut self,
        price: f64,
        quantity: f64,
        side: Side,
    ) -> (u64, Vec<Trade>) {
        let order_id = self.next_order_id;
        self.next_order_id += 1;

        let order = Order::new(
            order_id,
            price,
            quantity,
            side,
            OrderType::Limit,
            self.timestamp,
        );
        let trades = self.match_order(order);

        (order_id, trades)
    }

    /** Submit a market order and return trades */
    pub fn submit_market_order(&mut self, quantity: f64, side: Side) -> (u64, Vec<Trade>) {
        let order_id = self.next_order_id;
        self.next_order_id += 1;

        // Use aggressive price for market orders
        let price = match side {
            Side::Bid => f64::MAX,
            Side::Ask => 0.0,
        };

        let order = Order::new(
            order_id,
            price,
            quantity,
            side,
            OrderType::Market,
            self.timestamp,
        );
        let trades = self.match_order(order);

        (order_id, trades)
    }

    /** Match an incoming order against the book */
    fn match_order(&mut self, mut order: Order) -> Vec<Trade> {
        let mut trades = Vec::new();

        match order.side {
            Side::Bid => {
                // Match against asks (ascending price order)
                let mut sorted_ask_keys: Vec<_> = self.asks.keys().copied().collect();
                sorted_ask_keys.sort();

                for ask_price_key in sorted_ask_keys {
                    if order.is_filled() {
                        break;
                    }

                    let ask_price = ask_price_key as f64 / self.price_precision;

                    // Stop if order price is below ask price (for limit orders)
                    if order.order_type == OrderType::Limit && order.price < ask_price {
                        break;
                    }

                    if let Some(level) = self.asks.get_mut(&ask_price_key) {
                        while !level.is_empty() && !order.is_filled() {
                            if let Some(mut maker_order) = level.orders.front().cloned() {
                                let fill_qty = order.remaining().min(maker_order.remaining());

                                order.filled += fill_qty;
                                maker_order.filled += fill_qty;
                                level.total_quantity -= fill_qty;

                                trades.push(Trade {
                                    maker_order_id: maker_order.id,
                                    taker_order_id: order.id,
                                    price: ask_price,
                                    quantity: fill_qty,
                                    side: order.side,
                                    timestamp: self.timestamp,
                                });

                                if maker_order.is_filled() {
                                    level.orders.pop_front();
                                } else {
                                    // Update the front order
                                    if let Some(front) = level.orders.front_mut() {
                                        front.filled = maker_order.filled;
                                    }
                                }
                            }
                        }
                    }
                }

                // Remove empty price levels
                self.asks.retain(|_, level| !level.is_empty());

                // Add remaining to book if limit order
                if order.order_type == OrderType::Limit && !order.is_filled() {
                    let price_key = self.price_to_key(order.price);
                    let level = self
                        .bids
                        .entry(price_key)
                        .or_insert_with(|| PriceLevel::new(order.price));
                    level.add_order(order);
                }
            }
            Side::Ask => {
                // Match against bids (descending price order)
                let mut sorted_bid_keys: Vec<_> = self.bids.keys().copied().collect();
                sorted_bid_keys.sort_by(|a, b| b.cmp(a));

                for bid_price_key in sorted_bid_keys {
                    if order.is_filled() {
                        break;
                    }

                    let bid_price = bid_price_key as f64 / self.price_precision;

                    // Stop if order price is above bid price (for limit orders)
                    if order.order_type == OrderType::Limit && order.price > bid_price {
                        break;
                    }

                    if let Some(level) = self.bids.get_mut(&bid_price_key) {
                        while !level.is_empty() && !order.is_filled() {
                            if let Some(mut maker_order) = level.orders.front().cloned() {
                                let fill_qty = order.remaining().min(maker_order.remaining());

                                order.filled += fill_qty;
                                maker_order.filled += fill_qty;
                                level.total_quantity -= fill_qty;

                                trades.push(Trade {
                                    maker_order_id: maker_order.id,
                                    taker_order_id: order.id,
                                    price: bid_price,
                                    quantity: fill_qty,
                                    side: order.side,
                                    timestamp: self.timestamp,
                                });

                                if maker_order.is_filled() {
                                    level.orders.pop_front();
                                } else {
                                    if let Some(front) = level.orders.front_mut() {
                                        front.filled = maker_order.filled;
                                    }
                                }
                            }
                        }
                    }
                }

                // Remove empty price levels
                self.bids.retain(|_, level| !level.is_empty());

                // Add remaining to book if limit order
                if order.order_type == OrderType::Limit && !order.is_filled() {
                    let price_key = self.price_to_key(order.price);
                    let level = self
                        .asks
                        .entry(price_key)
                        .or_insert_with(|| PriceLevel::new(order.price));
                    level.add_order(order);
                }
            }
        }

        trades
    }

    /** Cancel an order by ID */
    pub fn cancel_order(&mut self, order_id: u64) -> bool {
        // Search in bids
        for level in self.bids.values_mut() {
            if let Some(pos) = level.orders.iter().position(|o| o.id == order_id) {
                let order = level.orders.remove(pos).unwrap();
                level.total_quantity -= order.remaining();
                // Clean up empty levels
                self.bids.retain(|_, level| !level.is_empty());
                return true;
            }
        }

        // Search in asks
        for level in self.asks.values_mut() {
            if let Some(pos) = level.orders.iter().position(|o| o.id == order_id) {
                let order = level.orders.remove(pos).unwrap();
                level.total_quantity -= order.remaining();
                // Clean up empty levels
                self.asks.retain(|_, level| !level.is_empty());
                return true;
            }
        }

        false
    }

    /** Clear the entire order book */
    pub fn clear(&mut self) {
        self.bids.clear();
        self.asks.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_order_book_creation() {
        let book = OrderBook::new();
        assert!(book.best_bid().is_none());
        assert!(book.best_ask().is_none());
    }

    #[test]
    fn test_limit_order_submission() {
        let mut book = OrderBook::new();

        // Add some bids
        book.submit_limit_order(100.0, 10.0, Side::Bid);
        book.submit_limit_order(99.0, 20.0, Side::Bid);

        // Add some asks
        book.submit_limit_order(101.0, 15.0, Side::Ask);
        book.submit_limit_order(102.0, 25.0, Side::Ask);

        assert_eq!(book.best_bid(), Some(100.0));
        assert_eq!(book.best_ask(), Some(101.0));
        assert_eq!(book.spread(), Some(1.0));
    }

    #[test]
    fn test_order_matching() {
        let mut book = OrderBook::new();

        // Add a limit ask
        book.submit_limit_order(100.0, 10.0, Side::Ask);

        // Submit a market buy that should match
        let (_, trades) = book.submit_market_order(5.0, Side::Bid);

        assert_eq!(trades.len(), 1);
        assert_eq!(trades[0].quantity, 5.0);
        assert_eq!(trades[0].price, 100.0);
    }

    #[test]
    fn test_order_cancellation() {
        let mut book = OrderBook::new();

        let (order_id, _) = book.submit_limit_order(100.0, 10.0, Side::Bid);
        assert!(book.best_bid().is_some());

        let cancelled = book.cancel_order(order_id);
        assert!(cancelled);
        assert!(book.best_bid().is_none());
    }

    #[test]
    fn test_imbalance() {
        let mut book = OrderBook::new();

        book.submit_limit_order(100.0, 100.0, Side::Bid);
        book.submit_limit_order(101.0, 50.0, Side::Ask);

        let imbalance = book.imbalance();
        // (100 - 50) / (100 + 50) = 50/150 ≈ 0.333
        assert!((imbalance - 0.333).abs() < 0.01);
    }
}
