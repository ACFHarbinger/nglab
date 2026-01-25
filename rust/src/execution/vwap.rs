/*!
 * Volume-Weighted Average Price (VWAP) execution algorithm.
 */

use crate::simulation::orderbook::{OrderBook, Side};
use serde::{Deserialize, Serialize};
use ts_rs::TS;

#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[ts(export)]
/// Execution state for a VWAP order.
pub struct VwapState {
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
    /// Historical or expected volume profile (normalized).
    pub volume_profile: Vec<f64>,
}

impl VwapState {
    /// Create a new VWAP execution state.
    pub fn new(
        quantity: f64,
        side: Side,
        start_step: u64,
        duration: u64,
        profile: Option<Vec<f64>>,
    ) -> Self {
        let end_step = start_step + duration;
        let volume_profile = profile.unwrap_or_else(|| vec![1.0; duration as usize]); // Default to uniform

        Self {
            total_quantity: quantity,
            executed_quantity: 0.0,
            start_step,
            end_step,
            side,
            volume_profile,
        }
    }

    /// Perform one execution step for the VWAP order based on the volume profile.
    pub fn step(&mut self, current_step: u64, orderbook: &mut OrderBook) {
        if current_step < self.start_step || current_step >= self.end_step {
            return;
        }

        let step_idx = (current_step - self.start_step) as usize;
        if step_idx >= self.volume_profile.len() {
            return;
        }

        let total_expected_vol: f64 = self.volume_profile.iter().sum();
        let current_step_vol = self.volume_profile[step_idx];

        let target_qty = (current_step_vol / total_expected_vol) * self.total_quantity;
        let remaining_qty = self.total_quantity - self.executed_quantity;
        let slice_qty = target_qty.min(remaining_qty).max(0.0);

        if slice_qty > 0.0 {
            if let Ok((_, trades)) = orderbook.submit_market_order(slice_qty, self.side) {
                for trade in trades {
                    self.executed_quantity += trade.quantity;
                }
            }
        }
    }

    /// Check if the VWAP order is finished.
    pub fn is_finished(&self, current_step: u64) -> bool {
        current_step >= self.end_step || self.executed_quantity >= self.total_quantity
    }
}
