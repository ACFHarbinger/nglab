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
use std::collections::{HashSet, VecDeque};
use ts_rs::TS;

use crate::errors::ArenaResult;
use crate::validation::{validate_price, validate_quantity};

/**
 * Order side: Bid (buy) or Ask (sell).
 */
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, TS)]
#[cfg_attr(feature = "python", pyclass(get_all))]
#[ts(export)]
pub enum Side {
    /// Buy side.
    Bid,
    /// Sell side.
    Ask,
}

impl Side {
    /// Get the opposite side.
    pub fn opposite(&self) -> Self {
        match self {
            Side::Bid => Side::Ask,
            Side::Ask => Side::Bid,
        }
    }
}

/**
 * Reference price for pegged orders.
 */
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, TS)]
#[cfg_attr(feature = "python", pyclass(get_all))]
#[ts(export)]
pub enum PegReference {
    /// Peg to the best bid.
    BestBid,
    /// Peg to the best ask.
    BestAsk,
    /// Peg to the mid point.
    MidPoint,
}

/**
 * Supported order types.
 */
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, TS)]
#[cfg_attr(feature = "python", pyclass)]
#[ts(export)]
pub enum OrderType {
    /// Limit order to buy/sell at specific price.
    Limit,
    /// Market order to execute immediately.
    Market,
    /// Stop order triggering a market order.
    StopLoss,
    /// Take profit order triggering a market order.
    TakeProfit,
    /// Stop limit order triggering a limit order.
    StopLimit,
    /// Fill the entire order immediately or cancel it.
    FillOrKill,
    /// Execute what's available immediately, cancel the rest.
    ImmediateOrCancel,
    /// Pegged order tracking a reference price.
    Pegged,
}

/**
 * Auction phases for market microstructure simulation.
 */
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum AuctionPhase {
    /// Normal continuous trading.
    #[default]
    None,
    /// Opening auction phase.
    Opening,
    /// Closing auction phase.
    Closing,
    /// Volatility auction phase (triggered by circuit breakers).
    Volatility,
}

/**
 * State for managing orders during an auction.
 */
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AuctionState {
    /// Current auction phase
    pub phase: AuctionPhase,
    /// Collected orders for the auction
    pub orders: Vec<Order>,
}

impl AuctionState {
    /// Creates a new AuctionState with the given phase.
    pub fn new(phase: AuctionPhase) -> Self {
        AuctionState {
            phase,
            orders: Vec::new(),
        }
    }
}

/**
 * A single order entry in the book.
 */
#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[cfg_attr(feature = "python", pyclass(get_all))]
#[ts(export)]
pub struct Order {
    /// Unique identifier for the order
    pub id: u64,
    /// Price per unit
    pub price: f64,
    /// Total quantity to be traded
    pub quantity: f64,
    /// Quantity already filled
    pub filled: f64,
    /// Side of the market (Bid or Ask)
    pub side: Side,
    /// Type of order (Limit, Market, etc.)
    pub order_type: OrderType,
    /// Unix timestamp of order submission
    pub timestamp: u64,
    /// Price that triggers the order (for Stop/TakeProfit)
    pub trigger_price: Option<f64>,
    /// Total quantity for Iceberg orders
    pub iceberg_total: Option<f64>,
    /// Currently visible quantity for Iceberg orders
    pub visible_quantity: Option<f64>,
    /// Identifier of the parent order (for OCO)
    pub parent_id: Option<u64>,
    /// Delta for trailing stop orders
    pub trailing_delta: Option<f64>,
    /// Current trigger point for trailing stop orders
    pub current_trailing_price: Option<f64>,
    /// Unix timestamp when order expires
    pub expiry: Option<u64>,
    /// Sibling order ID for One-Cancels-Other (OCO)
    pub oco_id: Option<u64>,
    /// Stop loss price for bracket order
    pub bracket_sl: Option<f64>,
    /// Take profit price for bracket order
    pub bracket_tp: Option<f64>,
    /// Reference for pegged orders
    pub peg_reference: Option<PegReference>,
    /// Offset for pegged orders
    pub peg_offset: Option<f64>,
}

/**
 * An order that has been submitted but not yet processed due to latency.
 */
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingOrder {
    /// The actual order
    pub order: Order,
    /// The timestamp when this order should be processed
    pub process_at: u64,
}

impl Order {
    /// Creates a new basic limit or market order.
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
            trailing_delta: None,
            current_trailing_price: None,
            expiry: None,
            oco_id: None,
            bracket_sl: None,
            bracket_tp: None,
            peg_reference: None,
            peg_offset: None,
        }
    }

    /// Creates a new advanced order with trigger, iceberg, or OCO fields.
    #[allow(clippy::too_many_arguments)]
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
        trailing_delta: Option<f64>,
        expiry: Option<u64>,
        oco_id: Option<u64>,
        bracket_sl: Option<f64>,
        bracket_tp: Option<f64>,
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
            trailing_delta,
            current_trailing_price: None,
            expiry,
            oco_id,
            bracket_sl,
            bracket_tp,
            peg_reference: None,
            peg_offset: None,
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
            let remaining_total = (total - self.filled).max(0.0);
            if remaining_total <= 0.0000001 {
                self.iceberg_total = Some(0.0);
                self.quantity = 0.0;
                self.filled = 0.0;
            } else {
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
    /// The price of this level
    pub price: f64,
    /// FIFO queue of orders at this price
    pub orders: VecDeque<Order>,
    /// Aggregated quantity of all orders in the queue
    pub total_quantity: f64,
}

impl PriceLevel {
    /// Creates a new empty price level.
    pub fn new(price: f64) -> Self {
        PriceLevel {
            price,
            orders: VecDeque::new(),
            total_quantity: 0.0,
        }
    }

    /** Add a target order to this price level */
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
#[cfg_attr(feature = "python", pyclass(get_all))]
pub struct Trade {
    /// ID of the maker order (the order already in the book)
    pub maker_order_id: u64,
    /// ID of the taker order (the incoming order)
    pub taker_order_id: u64,
    /// Execution price
    pub price: f64,
    /// Transacted quantity
    pub quantity: f64,
    /// Side of the taker order
    pub side: Side,
    /// Unix timestamp of execution
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
    pub stop_orders: Vec<Order>,
    /** Next order ID */
    next_order_id: u64,
    /** Current timestamp */
    timestamp: u64,
    /** Price precision multiplier */
    price_precision: f64,
    /** Last trade price for trigger checking */
    pub last_price: f64,
    /** Minimum price increment */
    tick_size: f64,
    /** State for the current auction, if any */
    auction_state: Option<AuctionState>,
    /** Orders waiting for latency to pass */
    pending_orders: VecDeque<PendingOrder>,
    /** Simulated network/processing latency in milliseconds */
    latency_ms: u64,
    /** Set of active pegged order IDs for efficient updates */
    pub pegged_orders: HashSet<u64>,
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

    #[pyo3(name = "modify_order")]
    pub fn modify_order_py(
        &mut self,
        order_id: u64,
        new_price: f64,
        new_quantity: f64,
        timestamp: u64,
    ) -> Option<u64> {
        self.modify_order(order_id, new_price, new_quantity, timestamp)
    }

    #[pyo3(name = "submit_fok_order")]
    pub fn submit_fok_order_py(&mut self, price: f64, quantity: f64, side: Side) -> Option<u64> {
        self.submit_fok_order(price, quantity, side)
            .ok()
            .map(|(id, _)| id)
    }

    #[pyo3(name = "submit_ioc_order")]
    pub fn submit_ioc_order_py(&mut self, price: f64, quantity: f64, side: Side) -> Option<u64> {
        self.submit_ioc_order(price, quantity, side)
            .ok()
            .map(|(id, _)| id)
    }

    #[pyo3(name = "prune_expired_orders")]
    pub fn prune_expired_orders_py(&mut self, current_timestamp: u64) {
        self.prune_expired_orders(current_timestamp);
    }

    #[pyo3(name = "submit_bracket_order")]
    pub fn submit_bracket_order_py(
        &mut self,
        price: f64,
        quantity: f64,
        side: Side,
        sl_price: f64,
        tp_price: f64,
    ) -> u64 {
        self.submit_bracket_order(price, quantity, side, sl_price, tp_price)
    }

    #[pyo3(name = "submit_pegged_order")]
    pub fn submit_pegged_order_py(
        &mut self,
        quantity: f64,
        side: Side,
        peg_reference: PegReference,
        peg_offset: f64,
    ) -> Option<u64> {
        self.submit_pegged_order(quantity, side, peg_reference, peg_offset)
    }
}

// =========================================================================
// Pure Rust Implementation
// =========================================================================

impl OrderBook {
    /// Create a new empty OrderBook.
    pub fn new() -> Self {
        OrderBook {
            bids: IndexMap::new(),
            asks: IndexMap::new(),
            stop_orders: Vec::new(),
            next_order_id: 1,
            timestamp: 0,
            price_precision: 10000.0,
            last_price: 100.0,
            tick_size: 0.01,
            auction_state: None,
            pending_orders: VecDeque::new(),
            latency_ms: 0,
            pegged_orders: HashSet::new(),
        }
    }

    /// Set simulated network latency in milliseconds.
    pub fn set_latency(&mut self, latency_ms: u64) {
        self.latency_ms = latency_ms;
    }

    /// Process pending orders that have overcome latency.
    pub fn process_latency(&mut self, current_time: u64) -> Vec<Trade> {
        let mut trades = Vec::new();
        while let Some(pending) = self.pending_orders.front() {
            if pending.process_at <= current_time {
                let pending = self.pending_orders.pop_front().unwrap();
                trades.extend(self.process_incoming_order(pending.order));
            } else {
                break;
            }
        }
        trades
    }

    fn process_incoming_order(&mut self, order: Order) -> Vec<Trade> {
        if matches!(
            order.order_type,
            OrderType::StopLoss | OrderType::TakeProfit | OrderType::StopLimit
        ) {
            self.stop_orders.push(order);
            Vec::new()
        } else {
            let trades = self.match_order(order);
            if !self.pegged_orders.is_empty() {
                self.reprice_pegged_orders();
            }
            trades
        }
    }

    /// Set the tick size for price rounding.
    pub fn set_tick_size(&mut self, tick_size: f64) {
        self.tick_size = tick_size;
    }

    /// Start an auction phase.
    pub fn begin_auction(&mut self, phase: AuctionPhase) {
        self.auction_state = Some(AuctionState::new(phase));
    }

    /// End the auction and clear orders.
    pub fn end_auction(&mut self) -> Vec<Trade> {
        let mut trades = Vec::new();
        if let Some(state) = self.auction_state.take() {
            if let Some(clearing_price) = self.calculate_clearing_price(&state.orders) {
                for mut order in state.orders {
                    if (order.side == Side::Bid && order.price >= clearing_price)
                        || (order.side == Side::Ask && order.price <= clearing_price)
                    {
                        order.order_type = OrderType::Limit;
                    }
                    trades.extend(self.match_order(order));
                }
            }
        }
        trades
    }

    /// Calculate the price that maximizes traded volume.
    pub fn calculate_clearing_price(&self, orders: &[Order]) -> Option<f64> {
        if orders.is_empty() {
            return None;
        }
        let mut prices: Vec<f64> = orders.iter().map(|o| o.price).collect();
        prices.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        prices.dedup();
        let mut best_price = None;
        let mut max_volume = 0.0;
        for &p in &prices {
            let mut bid_vol = 0.0;
            let mut ask_vol = 0.0;
            for o in orders {
                if o.side == Side::Bid && o.price >= p {
                    bid_vol += o.remaining();
                } else if o.side == Side::Ask && o.price <= p {
                    ask_vol += o.remaining();
                }
            }
            let vol = bid_vol.min(ask_vol);
            if vol > max_volume {
                max_volume = vol;
                best_price = Some(p);
            }
        }
        best_price
    }

    /// Get the position of an order in the queue.
    pub fn get_queue_position(&self, order_id: u64) -> Option<usize> {
        for level in self.bids.values() {
            if let Some(pos) = level.orders.iter().position(|o| o.id == order_id) {
                return Some(pos);
            }
        }
        for level in self.asks.values() {
            if let Some(pos) = level.orders.iter().position(|o| o.id == order_id) {
                return Some(pos);
            }
        }
        None
    }

    /// Submit an order with advanced parameters (Iceberg, OCO, Bracket, etc).
    #[allow(clippy::too_many_arguments)]
    pub fn submit_advanced_order(
        &mut self,
        price: f64,
        quantity: f64,
        side: Side,
        order_type: OrderType,
        trigger_price: Option<f64>,
        iceberg_total: Option<f64>,
        visible_quantity: Option<f64>,
        parent_id: Option<u64>,
        trailing_delta: Option<f64>,
        expiry: Option<u64>,
        oco_id: Option<u64>,
        bracket_sl: Option<f64>,
        bracket_tp: Option<f64>,
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
            visible_quantity,
            parent_id,
            trailing_delta,
            expiry,
            oco_id,
            bracket_sl,
            bracket_tp,
        );
        if let Some(state) = &mut self.auction_state {
            state.orders.push(order);
        } else if self.latency_ms > 0 {
            self.pending_orders.push_back(PendingOrder {
                process_at: self.timestamp + self.latency_ms,
                order,
            });
        } else {
            self.process_incoming_order(order);
        }
        order_id
    }

    /// Check and trigger stop/trailing orders based on current price.
    pub fn check_triggers(&mut self, current_price: f64) -> Vec<Trade> {
        let mut triggered_trades = Vec::new();
        let mut activated_orders = Vec::new();
        self.last_price = current_price;
        let mut i = 0;
        while i < self.stop_orders.len() {
            let mut order = self.stop_orders.remove(i);
            if let Some(delta) = order.trailing_delta {
                let current_tp = order.current_trailing_price.unwrap_or(match order.side {
                    Side::Bid => current_price + delta,
                    Side::Ask => current_price - delta,
                });
                let new_tp = match order.side {
                    Side::Bid => (current_price + delta).min(current_tp),
                    Side::Ask => (current_price - delta).max(current_tp),
                };
                order.current_trailing_price = Some(new_tp);
                order.trigger_price = Some(new_tp);
            }
            let should_trigger = match (order.side, order.trigger_price) {
                (Side::Bid, Some(tp)) => current_price >= tp,
                (Side::Ask, Some(tp)) => current_price <= tp,
                _ => false,
            };
            if should_trigger {
                order.order_type = match order.order_type {
                    OrderType::StopLimit => OrderType::Limit,
                    _ => OrderType::Market,
                };
                activated_orders.push(order);
            } else {
                self.stop_orders.insert(i, order);
                i += 1;
            }
        }
        for order in activated_orders {
            triggered_trades.extend(self.match_order(order));
        }
        triggered_trades
    }

    /// Get the best bid price.
    pub fn best_bid(&self) -> Option<f64> {
        self.bids
            .keys()
            .max()
            .map(|&p| p as f64 / self.price_precision)
    }

    /// Get the best ask price.
    pub fn best_ask(&self) -> Option<f64> {
        self.asks
            .keys()
            .min()
            .map(|&p| p as f64 / self.price_precision)
    }

    /// Get the mid price between best bid and best ask.
    pub fn mid_price(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some((bid + ask) / 2.0),
            _ => None,
        }
    }

    /// Get the spread (ask - bid).
    pub fn spread(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some(ask - bid),
            _ => None,
        }
    }

    /// Get the bid side depth up to `levels`.
    pub fn bid_depth(&self, levels: usize) -> Vec<(f64, f64)> {
        let mut keys: Vec<_> = self.bids.keys().copied().collect();
        keys.sort_by(|a, b| b.cmp(a));
        keys.into_iter()
            .take(levels)
            .map(|k| {
                (
                    k as f64 / self.price_precision,
                    self.bids[&k].total_quantity,
                )
            })
            .collect()
    }

    /// Get the ask side depth up to `levels`.
    pub fn ask_depth(&self, levels: usize) -> Vec<(f64, f64)> {
        let mut keys: Vec<_> = self.asks.keys().copied().collect();
        keys.sort();
        keys.into_iter()
            .take(levels)
            .map(|k| {
                (
                    k as f64 / self.price_precision,
                    self.asks[&k].total_quantity,
                )
            })
            .collect()
    }

    /// Get total volume on the bid side.
    pub fn total_bid_volume(&self) -> f64 {
        self.bids.values().map(|l| l.total_quantity).sum()
    }

    /// Get total volume on the ask side.
    pub fn total_ask_volume(&self) -> f64 {
        self.asks.values().map(|l| l.total_quantity).sum()
    }

    /// Calculate order book imbalance.
    pub fn imbalance(&self) -> f64 {
        let (b, a) = (self.total_bid_volume(), self.total_ask_volume());
        if b + a == 0.0 {
            0.0
        } else {
            (b - a) / (b + a)
        }
    }

    /// Update the current timestamp of the order book.
    pub fn set_timestamp(&mut self, timestamp: u64) {
        self.timestamp = timestamp;
    }

    /// Remove orders that have expired.
    pub fn prune_expired_orders(&mut self, current_timestamp: u64) {
        let prune = |levels: &mut IndexMap<i64, PriceLevel>| {
            for level in levels.values_mut() {
                level
                    .orders
                    .retain(|o| o.expiry.map_or(true, |e| e > current_timestamp));
                level.total_quantity = level.orders.iter().map(|o| o.remaining()).sum();
            }
            levels.retain(|_, l| !l.is_empty());
        };
        prune(&mut self.bids);
        prune(&mut self.asks);
        self.stop_orders
            .retain(|o| o.expiry.map_or(true, |e| e > current_timestamp));
    }

    fn cancel_oco_sibling(&mut self, oco_id: u64, current_order_id: u64) {
        let mut target_id = None;
        for level in self.bids.values() {
            if let Some(o) = level
                .orders
                .iter()
                .find(|o| o.oco_id == Some(oco_id) && o.id != current_order_id)
            {
                target_id = Some(o.id);
                break;
            }
        }
        if target_id.is_none() {
            for level in self.asks.values() {
                if let Some(o) = level
                    .orders
                    .iter()
                    .find(|o| o.oco_id == Some(oco_id) && o.id != current_order_id)
                {
                    target_id = Some(o.id);
                    break;
                }
            }
        }
        if target_id.is_none() {
            if let Some(o) = self
                .stop_orders
                .iter()
                .find(|o| o.oco_id == Some(oco_id) && o.id != current_order_id)
            {
                target_id = Some(o.id);
            }
        }
        if let Some(id) = target_id {
            self.cancel_order(id);
        }
    }

    /// Insert an order directly into the book (internal helper).
    pub fn submit_order_to_book(&mut self, order: Order) {
        let key = self.price_to_key(order.price);
        match order.side {
            Side::Bid => self
                .bids
                .entry(key)
                .or_insert_with(|| PriceLevel::new(order.price))
                .add_order(order),
            Side::Ask => self
                .asks
                .entry(key)
                .or_insert_with(|| PriceLevel::new(order.price))
                .add_order(order),
        }
    }
}

impl OrderBook {
    fn price_to_key(&self, price: f64) -> i64 {
        (price * self.price_precision).round() as i64
    }

    /// Submit a Limit order.
    pub fn submit_limit_order(
        &mut self,
        price: f64,
        quantity: f64,
        side: Side,
    ) -> ArenaResult<(u64, Vec<Trade>)> {
        validate_price(price)?;
        validate_quantity(quantity, None)?;
        let rounded = (price / self.tick_size).round() * self.tick_size;
        let id = self.next_order_id;
        self.next_order_id += 1;
        let order = Order::new(
            id,
            rounded,
            quantity,
            side,
            OrderType::Limit,
            self.timestamp,
        );
        Ok((id, self.match_order(order)))
    }

    /// Submit a Market order.
    pub fn submit_market_order(
        &mut self,
        quantity: f64,
        side: Side,
    ) -> ArenaResult<(u64, Vec<Trade>)> {
        validate_quantity(quantity, None)?;
        let id = self.next_order_id;
        self.next_order_id += 1;
        let price = if side == Side::Bid { f64::MAX } else { 0.0 };
        let order = Order::new(id, price, quantity, side, OrderType::Market, self.timestamp);
        Ok((id, self.match_order(order)))
    }

    /// Submit a Fill-or-Kill order.
    pub fn submit_fok_order(
        &mut self,
        price: f64,
        quantity: f64,
        side: Side,
    ) -> ArenaResult<(u64, Vec<Trade>)> {
        validate_price(price)?;
        validate_quantity(quantity, None)?;
        let id = self.next_order_id;
        self.next_order_id += 1;
        let order = Order::new(
            id,
            price,
            quantity,
            side,
            OrderType::FillOrKill,
            self.timestamp,
        );
        Ok((id, self.match_order(order)))
    }

    /// Submit an Immediate-or-Cancel order.
    pub fn submit_ioc_order(
        &mut self,
        price: f64,
        quantity: f64,
        side: Side,
    ) -> ArenaResult<(u64, Vec<Trade>)> {
        validate_price(price)?;
        validate_quantity(quantity, None)?;
        let id = self.next_order_id;
        self.next_order_id += 1;
        let order = Order::new(
            id,
            price,
            quantity,
            side,
            OrderType::ImmediateOrCancel,
            self.timestamp,
        );
        Ok((id, self.match_order(order)))
    }

    /// Submit a Bracket order (Limit entry + TP/SL).
    pub fn submit_bracket_order(
        &mut self,
        price: f64,
        quantity: f64,
        side: Side,
        sl_price: f64,
        tp_price: f64,
    ) -> u64 {
        self.submit_advanced_order(
            price,
            quantity,
            side,
            OrderType::Limit,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            Some(sl_price),
            Some(tp_price),
        )
    }

    fn handle_order_fill(
        &mut self,
        order_id: u64,
        side: Side,
        filled_qty: f64,
        is_fully_filled: bool,
        oco_id: Option<u64>,
        b_sl: Option<f64>,
        b_tp: Option<f64>,
    ) {
        if filled_qty <= 0.0 {
            return;
        }
        if is_fully_filled {
            if let Some(oco) = oco_id {
                self.cancel_oco_sibling(oco, order_id);
            }
        }
        if b_sl.is_some() || b_tp.is_some() {
            let exit_side = side.opposite();
            let exit_oco = oco_id.unwrap_or_else(|| {
                let nid = self.next_order_id;
                self.next_order_id += 1;
                nid
            });
            if let Some(p) = b_sl {
                let sid = self.next_order_id;
                self.next_order_id += 1;
                self.stop_orders.push(Order::new_advanced(
                    sid,
                    0.0,
                    filled_qty,
                    exit_side,
                    OrderType::StopLoss,
                    self.timestamp,
                    Some(p),
                    None,
                    None,
                    Some(order_id),
                    None,
                    None,
                    Some(exit_oco),
                    None,
                    None,
                ));
            }
            if let Some(p) = b_tp {
                let tid = self.next_order_id;
                self.next_order_id += 1;
                self.stop_orders.push(Order::new_advanced(
                    tid,
                    0.0,
                    filled_qty,
                    exit_side,
                    OrderType::TakeProfit,
                    self.timestamp,
                    Some(p),
                    None,
                    None,
                    Some(order_id),
                    None,
                    None,
                    Some(exit_oco),
                    None,
                    None,
                ));
            }
        }
    }

    fn match_order(&mut self, mut order: Order) -> Vec<Trade> {
        let mut trades = Vec::new();
        let mut maker_fills = Vec::new();
        let (id, side, oco_id, b_sl, b_tp) = (
            order.id,
            order.side,
            order.oco_id,
            order.bracket_sl,
            order.bracket_tp,
        );

        if order.order_type == OrderType::FillOrKill {
            let mut avail = 0.0;
            match side {
                Side::Bid => {
                    for (&k, l) in self.asks.iter() {
                        if (k as f64 / self.price_precision) <= order.price {
                            avail += l.total_quantity;
                        }
                    }
                }
                Side::Ask => {
                    let mut sorted_bid_keys: Vec<_> = self.bids.keys().copied().collect();
                    sorted_bid_keys.sort_by(|a, b| b.cmp(a));
                    for price_key in sorted_bid_keys {
                        let price = price_key as f64 / self.price_precision;
                        if price < order.price {
                            break;
                        }
                        if let Some(level) = self.bids.get(&price_key) {
                            avail += level.total_quantity;
                        }
                    }
                }
            }
            if avail < order.remaining() {
                return trades;
            }
        }

        let (final_filled, is_filled) = match side {
            Side::Bid => {
                let mut ask_keys: Vec<_> = self.asks.keys().copied().collect();
                ask_keys.sort();
                for k in ask_keys {
                    if order.is_filled() {
                        break;
                    }
                    let p = k as f64 / self.price_precision;
                    if matches!(
                        order.order_type,
                        OrderType::Limit
                            | OrderType::FillOrKill
                            | OrderType::ImmediateOrCancel
                            | OrderType::Pegged
                    ) && order.price < p
                    {
                        break;
                    }
                    if let Some(level) = self.asks.get_mut(&k) {
                        while !level.is_empty() && !order.is_filled() {
                            let mut maker = level.orders.front().cloned().unwrap();
                            if let Some(e) = maker.expiry {
                                if e <= self.timestamp {
                                    level.orders.pop_front();
                                    level.total_quantity -= maker.remaining();
                                    continue;
                                }
                            }
                            let qty = order.remaining().min(maker.remaining());
                            order.filled += qty;
                            maker.filled += qty;
                            level.total_quantity -= qty;
                            trades.push(Trade {
                                maker_order_id: maker.id,
                                taker_order_id: id,
                                price: p,
                                quantity: qty,
                                side: Side::Bid,
                                timestamp: self.timestamp,
                            });
                            maker_fills.push((
                                maker.id,
                                maker.side,
                                qty,
                                maker.is_filled(),
                                maker.oco_id,
                                maker.bracket_sl,
                                maker.bracket_tp,
                            ));
                            if maker.is_filled() {
                                level.orders.pop_front();
                                if maker.iceberg_total.is_some() {
                                    maker.refresh_iceberg();
                                    if maker.quantity > 0.0 {
                                        level.add_order(maker);
                                    }
                                }
                            } else if let Some(front) = level.orders.front_mut() {
                                front.filled = maker.filled;
                            }
                        }
                    }
                }
                self.asks.retain(|_, l| !l.is_empty());
                let (ff, iff) = (order.filled, order.is_filled());
                if (order.order_type == OrderType::Limit || order.order_type == OrderType::Pegged)
                    && !iff
                {
                    self.submit_order_to_book(order);
                }
                (ff, iff)
            }
            Side::Ask => {
                let mut bid_keys: Vec<_> = self.bids.keys().copied().collect();
                bid_keys.sort_by(|a, b| b.cmp(a));
                for k in bid_keys {
                    if order.is_filled() {
                        break;
                    }
                    let p = k as f64 / self.price_precision;
                    if matches!(
                        order.order_type,
                        OrderType::Limit
                            | OrderType::FillOrKill
                            | OrderType::ImmediateOrCancel
                            | OrderType::Pegged
                    ) && order.price > p
                    {
                        break;
                    }
                    if let Some(level) = self.bids.get_mut(&k) {
                        while !level.is_empty() && !order.is_filled() {
                            let mut maker = level.orders.front().cloned().unwrap();
                            if let Some(e) = maker.expiry {
                                if e <= self.timestamp {
                                    level.orders.pop_front();
                                    level.total_quantity -= maker.remaining();
                                    continue;
                                }
                            }
                            let qty = order.remaining().min(maker.remaining());
                            order.filled += qty;
                            maker.filled += qty;
                            level.total_quantity -= qty;
                            trades.push(Trade {
                                maker_order_id: maker.id,
                                taker_order_id: id,
                                price: p,
                                quantity: qty,
                                side: Side::Ask,
                                timestamp: self.timestamp,
                            });
                            maker_fills.push((
                                maker.id,
                                maker.side,
                                qty,
                                maker.is_filled(),
                                maker.oco_id,
                                maker.bracket_sl,
                                maker.bracket_tp,
                            ));
                            if maker.is_filled() {
                                level.orders.pop_front();
                                if maker.iceberg_total.is_some() {
                                    maker.refresh_iceberg();
                                    if maker.quantity > 0.0 {
                                        level.add_order(maker);
                                    }
                                }
                            } else if let Some(front) = level.orders.front_mut() {
                                front.filled = maker.filled;
                            }
                        }
                    }
                }
                self.bids.retain(|_, l| !l.is_empty());
                let (ff, iff) = (order.filled, order.is_filled());
                if (order.order_type == OrderType::Limit || order.order_type == OrderType::Pegged)
                    && !iff
                {
                    self.submit_order_to_book(order);
                }
                (ff, iff)
            }
        };

        for (m_id, m_side, m_qty, m_iff, m_oco, m_sl, m_tp) in maker_fills {
            self.handle_order_fill(m_id, m_side, m_qty, m_iff, m_oco, m_sl, m_tp);
        }
        self.handle_order_fill(id, side, final_filled, is_filled, oco_id, b_sl, b_tp);

        if !self.pegged_orders.is_empty() {
            self.reprice_pegged_orders();
        }

        trades
    }

    /// Cancel an order by ID. Returns true if found and cancelled.
    pub fn cancel_order(&mut self, order_id: u64) -> bool {
        let cancel = |levels: &mut IndexMap<i64, PriceLevel>| {
            for level in levels.values_mut() {
                if let Some(pos) = level.orders.iter().position(|o| o.id == order_id) {
                    let o = level.orders.remove(pos).unwrap();
                    level.total_quantity -= o.remaining();
                    return true;
                }
            }
            false
        };
        if cancel(&mut self.bids) || cancel(&mut self.asks) {
            self.bids.retain(|_, l| !l.is_empty());
            self.asks.retain(|_, l| !l.is_empty());
            true
        } else if let Some(pos) = self.stop_orders.iter().position(|o| o.id == order_id) {
            self.stop_orders.remove(pos);
            true
        } else {
            false
        }
    }

    /// Modify an existing order's price or quantity.
    pub fn modify_order(
        &mut self,
        order_id: u64,
        new_price: f64,
        new_quantity: f64,
        timestamp: u64,
    ) -> Option<u64> {
        self.timestamp = timestamp;
        let mut target = None;
        for level in self.bids.values() {
            if let Some(o) = level.orders.iter().find(|o| o.id == order_id) {
                target = Some(o.clone());
                break;
            }
        }
        if target.is_none() {
            for level in self.asks.values() {
                if let Some(o) = level.orders.iter().find(|o| o.id == order_id) {
                    target = Some(o.clone());
                    break;
                }
            }
        }
        if target.is_none() {
            if let Some(o) = self.stop_orders.iter().find(|o| o.id == order_id) {
                target = Some(o.clone());
            }
        }
        if let Some(o) = target {
            if o.price == new_price && new_quantity <= o.quantity {
                if self.cancel_order(order_id) {
                    let mut new_o = o.clone();
                    new_o.quantity = new_quantity;
                    self.submit_order_to_book(new_o);
                    return Some(order_id);
                }
            }
            self.cancel_order(order_id);
            return self
                .submit_limit_order(new_price, new_quantity, o.side)
                .ok()
                .map(|(id, _)| id);
        }
        None
    }

    /// Remove an order from the book and return it.
    pub fn pop_order(&mut self, order_id: u64) -> Option<Order> {
        let mut target = None;
        for (&price, level) in &self.bids {
            if let Some(idx) = level.orders.iter().position(|o| o.id == order_id) {
                target = Some((price, idx, Side::Bid));
                break;
            }
        }

        if target.is_none() {
            for (&price, level) in &self.asks {
                if let Some(idx) = level.orders.iter().position(|o| o.id == order_id) {
                    target = Some((price, idx, Side::Ask));
                    break;
                }
            }
        }

        if let Some((price, idx, side)) = target {
            let maps = match side {
                Side::Bid => &mut self.bids,
                Side::Ask => &mut self.asks,
            };

            if let Some(level) = maps.get_mut(&price) {
                if idx < level.orders.len() {
                    let order = level.orders.remove(idx).unwrap();
                    level.total_quantity -= order.remaining();

                    if level.is_empty() {
                        maps.swap_remove(&price);
                    }
                    return Some(order);
                }
            }
        }
        None
    }

    /// Clear all orders from the book.
    pub fn clear(&mut self) {
        self.bids.clear();
        self.asks.clear();
        self.stop_orders.clear();
        self.pegged_orders.clear();
    }

    /// Calculate peg price based on reference and offset.
    fn calculate_peg_price(&self, reference: PegReference, offset: f64) -> Option<f64> {
        let price = match reference {
            PegReference::BestBid => self.best_bid(),
            PegReference::BestAsk => self.best_ask(),
            PegReference::MidPoint => self.mid_price(),
        };

        // If reference is missing, default to last price or None
        let ref_price = price.or(Some(self.last_price)).filter(|&p| p > 0.0)?;

        let final_price = ref_price + offset;

        Some((final_price * self.price_precision).round() / self.price_precision)
    }

    /// Submit a pegged order.
    pub fn submit_pegged_order(
        &mut self,
        quantity: f64,
        side: Side,
        reference: PegReference,
        offset: f64,
    ) -> Option<u64> {
        let price = self.calculate_peg_price(reference, offset)?;

        let order_id = self.next_order_id;
        self.next_order_id += 1;

        let mut order = Order::new(
            order_id,
            price,
            quantity,
            side,
            OrderType::Pegged,
            self.timestamp,
        );
        order.peg_reference = Some(reference);
        order.peg_offset = Some(offset);

        self.pegged_orders.insert(order_id);

        if self.latency_ms > 0 {
            self.pending_orders.push_back(PendingOrder {
                process_at: self.timestamp + self.latency_ms,
                order,
            });
            Some(order_id)
        } else {
            self.process_incoming_order(order);
            Some(order_id)
        }
    }

    /// Reprice all active pegged orders based on current market data.
    pub fn reprice_pegged_orders(&mut self) {
        // Collect orders that need updates to avoid borrowing issues
        let mut updates = Vec::new();
        let mut to_remove = Vec::new();

        // Helper to find order info
        let find_order_info = |id: u64| -> Option<(f64, PegReference, f64)> {
            for level in self.bids.values() {
                if let Some(o) = level.orders.iter().find(|o| o.id == id) {
                    return Some((o.price, o.peg_reference?, o.peg_offset?));
                }
            }
            for level in self.asks.values() {
                if let Some(o) = level.orders.iter().find(|o| o.id == id) {
                    return Some((o.price, o.peg_reference?, o.peg_offset?));
                }
            }
            None
        };

        for &id in &self.pegged_orders {
            if let Some((current_price, reference, offset)) = find_order_info(id) {
                if let Some(new_price) = self.calculate_peg_price(reference, offset) {
                    if (new_price - current_price).abs() > std::f64::EPSILON {
                        updates.push((id, new_price));
                    }
                }
            } else {
                // Order not found (filled or cancelled), mark for removal
                to_remove.push(id);
            }
        }

        for id in to_remove {
            self.pegged_orders.remove(&id);
        }

        // Apply updates
        for (id, new_price) in updates {
            if let Some(mut order) = self.pop_order(id) {
                order.price = new_price;
                self.submit_order_to_book(order);
            }
        }
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
        book.submit_limit_order(100.0, 10.0, Side::Bid).unwrap();
        book.submit_limit_order(101.0, 15.0, Side::Ask).unwrap();
        assert_eq!(book.best_bid(), Some(100.0));
        assert_eq!(book.best_ask(), Some(101.0));
    }

    #[test]
    fn test_pegged_order_submission_and_update() {
        let mut book = OrderBook::new();
        book.submit_limit_order(100.0, 10.0, Side::Bid).unwrap();
        book.submit_limit_order(105.0, 10.0, Side::Ask).unwrap();

        // Peg to Best Bid + 0.0 (Primary Peg)
        // This avoids the self-referential infinite loop of aggressive pegs.
        let id = book
            .submit_pegged_order(5.0, Side::Bid, PegReference::BestBid, 0.0)
            .unwrap();

        // Initial check: Best Bid is 100.0. I join at 100.0.
        // My ID is tracked.
        assert_eq!(book.pegged_orders.len(), 1);
        assert!(book.pegged_orders.contains(&id));

        // Now update the market: New Limit Bid at 102.0
        // This establishes a NEW Best Bid at 102.0.
        // My order (currently at 100.0) should reprice to 102.0 logic:
        // Match runs -> updates BBO -> calls reprice_pegged_orders -> I move to 102.0.
        book.submit_limit_order(102.0, 10.0, Side::Bid).unwrap();

        // Verify my order is now at 102.0
        let order_info = book.get_queue_position(id);
        assert!(order_info.is_some());

        // Check price of the order in the book
        let mut found_price = None;
        for level in book.bids.values() {
            if level.orders.iter().any(|o| o.id == id) {
                found_price = Some(level.price);
                break;
            }
        }
        assert_eq!(found_price, Some(102.0));
    }

    #[test]
    fn test_order_matching() {
        let mut book = OrderBook::new();
        book.submit_limit_order(100.0, 10.0, Side::Ask).unwrap();
        let (_, trades) = book.submit_market_order(5.0, Side::Bid).unwrap();
        assert_eq!(trades.len(), 1);
        assert_eq!(trades[0].quantity, 5.0);
        assert_eq!(trades[0].price, 100.0);
    }

    #[test]
    fn test_order_cancellation() {
        let mut book = OrderBook::new();
        let (id, _) = book.submit_limit_order(100.0, 10.0, Side::Bid).unwrap();
        assert!(book.cancel_order(id));
        assert!(book.best_bid().is_none());
    }

    #[test]
    fn test_iceberg_execution() {
        let mut book = OrderBook::new();
        book.submit_advanced_order(
            101.0,
            10.0,
            Side::Ask,
            OrderType::Limit,
            None,
            Some(100.0),
            Some(10.0),
            None,
            None,
            None,
            None,
            None,
            None,
        );
        let (_, trades) = book.submit_market_order(10.0, Side::Bid).unwrap();
        assert_eq!(trades.len(), 1);
        assert_eq!(book.best_ask(), Some(101.0));
        assert_eq!(book.ask_depth(1)[0].1, 10.0);
    }

    #[test]
    fn test_trailing_stop_trigger() {
        let mut book = OrderBook::new();
        book.last_price = 100.0;
        let stop = Order::new_advanced(
            1,
            0.0,
            10.0,
            Side::Ask,
            OrderType::StopLoss,
            0,
            None,
            None,
            None,
            None,
            Some(5.0),
            None,
            None,
            None,
            None,
        );
        book.stop_orders.push(stop);
        book.check_triggers(110.0);
        assert_eq!(book.stop_orders[0].trigger_price, Some(105.0));
        book.submit_limit_order(100.0, 20.0, Side::Bid).unwrap();
        let trades = book.check_triggers(104.0);
        assert!(!trades.is_empty());
    }

    #[test]
    fn test_fok_execution() {
        let mut book = OrderBook::new();
        book.submit_limit_order(100.0, 10.0, Side::Ask).unwrap();
        let (_, trades) = book.submit_fok_order(100.0, 11.0, Side::Bid).unwrap();
        assert!(trades.is_empty());
    }

    #[test]
    fn test_ioc_execution() {
        let mut book = OrderBook::new();
        book.submit_limit_order(100.0, 10.0, Side::Ask).unwrap();
        let (_, trades) = book.submit_ioc_order(100.0, 15.0, Side::Bid).unwrap();
        assert_eq!(trades.len(), 1);
        assert_eq!(trades[0].quantity, 10.0);
        assert!(book.bids.is_empty());
    }

    #[test]
    fn test_bracket_execution() {
        let mut book = OrderBook::new();
        book.submit_bracket_order(100.0, 10.0, Side::Bid, 95.0, 105.0);
        book.submit_limit_order(100.0, 10.0, Side::Ask).unwrap();
        assert_eq!(book.stop_orders.len(), 2);
        book.submit_limit_order(90.0, 10.0, Side::Bid).unwrap();
        book.check_triggers(94.0);
        assert!(book.stop_orders.is_empty());
    }
}
