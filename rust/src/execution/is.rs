/*!
 * Implementation Shortfall (IS) execution algorithm.
 *
 * balances market impact cost (execution too fast) against timing risk (execution too slow).
 * Uses an urgency parameter to control the trade-off.
 */

use crate::simulation::orderbook::{OrderBook, Side, Trade};
use serde::{Deserialize, Serialize};
use ts_rs::TS;

#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[ts(export)]
/// Execution state for an Implementation Shortfall order.
pub struct IsState {
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
    /// Urgency parameter (0.0 = Risk Neutral/TWAP, 1.0 = High Urgency/Front-loaded).
    pub urgency: f64,
    /// Benchmark Arrival Price (Mid-price at start).
    pub arrival_price: Option<f64>,
}

impl IsState {
    /// Create a new IS execution state.
    pub fn new(quantity: f64, side: Side, start_step: u64, duration: u64, urgency: f64) -> Self {
        Self {
            total_quantity: quantity,
            executed_quantity: 0.0,
            start_step,
            end_step: start_step + duration,
            side,
            urgency: urgency.clamp(0.0, 10.0), // Allow >1.0 for very high urgency
            arrival_price: None,
        }
    }

    /// Perform one execution step for the IS order.
    pub fn step(&mut self, current_step: u64, orderbook: &mut OrderBook) -> Vec<Trade> {
        if current_step < self.start_step || current_step >= self.end_step {
            return Vec::new();
        }

        // Capture arrival price on first valid step
        if self.arrival_price.is_none() {
            self.arrival_price = orderbook.mid_price();
        }

        let remaining_steps = self.end_step - current_step;
        let remaining_qty = self.total_quantity - self.executed_quantity;

        if remaining_steps == 0 || remaining_qty <= 0.0 {
            return Vec::new();
        }

        // Implementation Shortfall Schedule Calculation
        // We use a simplified Almgren-Chriss style trajectory.
        // Optimal trajectory x(t) ~ sinh(k(T-t)) / sinh(kT)
        // Here we approximate with an exponential decay based on urgency.

        let time_horizon = (self.end_step - self.start_step) as f64;
        let time_remaining = remaining_steps as f64;

        // Kappa represents the decay rate derived from urgency
        // Higher urgency -> higher kappa -> faster initial execution
        let kappa = self.urgency * 2.0 + 1e-4; // Avoid 0

        // Calculate expected remaining quantity for this time step based on optimal curve
        // Target Remaining = Total * (sinh(k * rem_time) / sinh(k * total_time))
        let target_remaining_frac =
            (kappa * (time_remaining / time_horizon)).sinh() / (kappa).sinh();

        let target_remaining_qty = self.total_quantity * target_remaining_frac;

        // We want to be at target_remaining_qty by end of this step.
        // Current remaining is `remaining_qty`.
        // So we need to sell `remaining_qty - target_remaining_qty`.

        // However, we must ensure we don't buy negative amounts (if we are ahead of schedule)
        // In discrete steps, we aim to close the gap.

        let mut slice_qty = remaining_qty - target_remaining_qty;

        // Adaptive Component:
        // If price moved in our favor (Buy lower than arrival, Sell higher than arrival),
        // we can be more aggressive (opportunistic).
        if let Some(arrival) = self.arrival_price {
            if let Some(current_mid) = orderbook.mid_price() {
                let favorable_move = match self.side {
                    Side::Bid => (arrival - current_mid) / arrival,
                    Side::Ask => (current_mid - arrival) / arrival,
                };

                if favorable_move > 0.001 {
                    // > 0.1% improvement
                    // Increase slice by factor of favorability
                    slice_qty *= 1.0 + (favorable_move * 10.0);
                }
            }
        }

        let slice_qty = slice_qty.min(remaining_qty).max(0.0);

        if slice_qty > 0.0 {
            if let Ok((_, trades)) = orderbook.submit_market_order(slice_qty, self.side) {
                for trade in &trades {
                    self.executed_quantity += trade.quantity;
                }
                return trades;
            }
        }
        Vec::new()
    }

    /// Check if the IS order is finished.
    pub fn is_finished(&self, current_step: u64) -> bool {
        current_step >= self.end_step || self.executed_quantity >= self.total_quantity
    }
}
