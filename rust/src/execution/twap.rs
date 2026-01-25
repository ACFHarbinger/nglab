/*!
 * Time-Weighted Average Price (TWAP) execution algorithm.
 */

use crate::simulation::orderbook::{OrderBook, Side};
use serde::{Deserialize, Serialize};
use ts_rs::TS;

#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[ts(export)]
/// Execution state for a TWAP order.
pub struct TwapState {
    /// Total quantity to be executed.
    pub total_quantity: f64,
    /// Quantity executed so far.
    pub executed_quantity: f64,
    /// Simulation step when the order started.
    pub start_step: u64,
    /// Simulation step when the order should end.
    pub end_step: u64,
    /// Side of the market (Bid/Ask).
    pub side: Side,
}

impl TwapState {
    /// Create a new TWAP execution state.
    pub fn new(quantity: f64, side: Side, start_step: u64, duration: u64) -> Self {
        Self {
            total_quantity: quantity,
            executed_quantity: 0.0,
            start_step,
            end_step: start_step + duration,
            side,
        }
    }

    /// Perform one execution step for the TWAP order.
    pub fn step(&mut self, current_step: u64, orderbook: &mut OrderBook) {
        if current_step < self.start_step || current_step >= self.end_step {
            return;
        }

        let remaining_steps = self.end_step - current_step;
        let remaining_qty = self.total_quantity - self.executed_quantity;

        if remaining_steps == 0 || remaining_qty <= 0.0 {
            return;
        }

        // Random jitter to avoid detection - slice size varies slightly
        let base_slice = remaining_qty / remaining_steps as f64;
        let jitter = (rand::random::<f64>() - 0.5) * 0.2 * base_slice; // +/- 10% jitter
        let slice_qty = (base_slice + jitter).min(remaining_qty).max(0.0);

        if slice_qty > 0.0 {
            // Execute as Market Order for simplicity in this implementation
            if let Ok((_, trades)) = orderbook.submit_market_order(slice_qty, self.side) {
                for trade in trades {
                    self.executed_quantity += trade.quantity;
                }
            }
        }
    }

    /// Check if the TWAP order is finished.
    pub fn is_finished(&self, current_step: u64) -> bool {
        current_step >= self.end_step || self.executed_quantity >= self.total_quantity
    }
}
