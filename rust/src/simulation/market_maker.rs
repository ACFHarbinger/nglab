/*!
 * Market Maker Strategy
 *
 * Provides a framework for automated liquidity provision with inventory-based skew.
 */

use crate::simulation::orderbook::{OrderBook, Side};
use serde::{Deserialize, Serialize};

/// Configuration for the Market Maker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketMakerConfig {
    /// Desired spread between bid and ask (e.g., 0.01).
    pub target_spread: f64,
    /// Fixed quantity for each quote.
    pub quote_size: f64,
    /// Maximum absolute inventory allowed.
    pub max_inventory: f64,
    /// Intensity of price skew based on inventory (0.0 to 1.0).
    pub skew_intensity: f64,
}

impl Default for MarketMakerConfig {
    fn default() -> Self {
        Self {
            target_spread: 0.002,
            quote_size: 10.0,
            max_inventory: 100.0,
            skew_intensity: 0.5,
        }
    }
}

/// Internal state of the Market Maker.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MarketMakerState {
    /// ID of the active bid order.
    pub bid_id: Option<u64>,
    /// ID of the active ask order.
    pub ask_id: Option<u64>,
    /// Accumulated realized profit from spread capture.
    pub realized_pnl: f64,
}

/// Core Market Maker logic.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MarketMaker {
    /// Configuration settings.
    pub config: MarketMakerConfig,
    /// Runtime state.
    pub state: MarketMakerState,
    /// Whether the MM is currently active.
    pub active: bool,
}

impl MarketMaker {
    /// Create a new Market Maker with default config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Calculate the skewed bid and ask prices based on mid-price and inventory.
    pub fn calculate_quotes(&self, mid_price: f64, inventory: f64) -> (f64, f64) {
        let half_spread = self.config.target_spread / 2.0;

        // Calculate skew factor (-1.0 to 1.0)
        let skew_factor = (inventory / self.config.max_inventory).clamp(-1.0, 1.0);
        let skew_adjustment = half_spread * skew_factor * self.config.skew_intensity;

        // Skew prices:
        // If inventory is positive (Long), lower both bid and ask to encourage sell-off.
        // If inventory is negative (Short), raise both bid and ask to encourage buy-back.
        let bid_price = mid_price - half_spread - skew_adjustment;
        let ask_price = mid_price + half_spread - skew_adjustment;

        (bid_price, ask_price)
    }

    /// Update the quotes in the orderbook.
    pub fn update_quotes(&mut self, orderbook: &mut OrderBook, inventory: f64) {
        if !self.active {
            self.cancel_quotes(orderbook);
            return;
        }

        let mid_price = match orderbook.mid_price() {
            Some(p) => p,
            None => return,
        };

        let (target_bid, target_ask) = self.calculate_quotes(mid_price, inventory);

        // Update BID if significantly different or missing
        if self.should_reprice(self.state.bid_id, target_bid, orderbook) {
            if let Some(id) = self.state.bid_id {
                let _ = orderbook.cancel_order(id);
            }
            if let Ok((id, _)) =
                orderbook.submit_limit_order(target_bid, self.config.quote_size, Side::Bid)
            {
                self.state.bid_id = Some(id);
            }
        }

        // Update ASK if significantly different or missing
        if self.should_reprice(self.state.ask_id, target_ask, orderbook) {
            if let Some(id) = self.state.ask_id {
                let _ = orderbook.cancel_order(id);
            }
            if let Ok((id, _)) =
                orderbook.submit_limit_order(target_ask, self.config.quote_size, Side::Ask)
            {
                self.state.ask_id = Some(id);
            }
        }
    }

    fn should_reprice(
        &self,
        order_id: Option<u64>,
        target_price: f64,
        orderbook: &OrderBook,
    ) -> bool {
        match order_id {
            None => true,
            Some(id) => {
                // If order is gone, reprice
                if !orderbook.has_order(id) {
                    return true;
                }
                // If price drifted more than 10% of the spread, reprice
                let threshold = self.config.target_spread * 0.1;
                let current_price = orderbook.get_order_price(id).unwrap_or(0.0);
                (current_price - target_price).abs() > threshold
            }
        }
    }

    /// Check for adverse selection conditions and cancel quotes if necessary.
    pub fn check_adverse_selection(&mut self, orderbook: &mut OrderBook) {
        // Simple metric: If order flow imbalance is too high against us, pull quotes.
        let imbalance = orderbook.imbalance(); // -1.0 to 1.0
        let threshold = 0.8; // Configurable?

        // If we are quoting both sides, high imbalance is bad.
        if imbalance.abs() > threshold {
            self.cancel_quotes(orderbook);
        }
    }

    /// Handle trade events to update P&L and Inventory.
    /// Note: This implies the caller must notify MM of its own trades.
    pub fn on_trade(&mut self, quantity: f64, price: f64, side: Side) {
        // Calculate P&L if this trade closes an existing position
        // For simplicity in this non-alloc implementation, we just track 'realized' as
        // spread capture approximation or raw cash flow.

        // Cash Flow method:
        // Buy: -Price * Qty
        // Sell: +Price * Qty
        let cash_flow = match side {
            Side::Bid => -price * quantity, // We bought
            Side::Ask => price * quantity,  // We sold
        };

        self.state.realized_pnl += cash_flow;
    }

    /// Cancel all active MM quotes.
    pub fn cancel_quotes(&mut self, orderbook: &mut OrderBook) {
        if let Some(id) = self.state.bid_id.take() {
            let _ = orderbook.cancel_order(id);
        }
        if let Some(id) = self.state.ask_id.take() {
            let _ = orderbook.cancel_order(id);
        }
    }
}
