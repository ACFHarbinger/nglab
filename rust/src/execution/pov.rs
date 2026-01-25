/*!
 * Percentage of Volume (POV) execution algorithm.
 */

use crate::simulation::orderbook::{OrderBook, Side};
use serde::{Deserialize, Serialize};
use ts_rs::TS;

#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[ts(export)]
/// Execution state for a Percentage of Volume (POV) order.
pub struct PovState {
    /// Total quantity to be executed.
    pub total_quantity: f64,
    /// Quantity executed so far.
    pub executed_quantity: f64,
    /// Side of the market (Bid/Ask).
    pub side: Side,
    /// Target participation rate (0.0 to 1.0).
    pub participation_rate: f64,
}

impl PovState {
    /// Create a new POV execution state.
    pub fn new(quantity: f64, side: Side, participation_rate: f64) -> Self {
        Self {
            total_quantity: quantity,
            executed_quantity: 0.0,
            side,
            participation_rate: participation_rate.min(1.0).max(0.0),
        }
    }

    /// React to market volume by submitting a portion to the orderbook.
    pub fn step(&mut self, orderbook: &mut OrderBook, market_volume: f64) {
        let remaining_qty = self.total_quantity - self.executed_quantity;
        if remaining_qty <= 0.0 || market_volume <= 0.0 {
            return;
        }

        // Target quantity is participation_rate of market volume
        let target_slice = market_volume * self.participation_rate;
        let slice_qty = target_slice.min(remaining_qty);

        if slice_qty > 0.0 {
            if let Ok((_, trades)) = orderbook.submit_market_order(slice_qty, self.side) {
                for trade in trades {
                    self.executed_quantity += trade.quantity;
                }
            }
        }
    }

    /// Check if the POV order is finished.
    pub fn is_finished(&self) -> bool {
        self.executed_quantity >= self.total_quantity
    }
}
