/*!
 * Algorithmic Execution Engine
 *
 * Provides institutional-grade execution algorithms like TWAP, VWAP, and POV.
 */

pub mod analytics;
pub mod is;
pub mod pov;
pub mod twap;
pub mod vwap;

use serde::{Deserialize, Serialize};
use ts_rs::TS;

#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[ts(export)]
/// Type of execution algorithm.
pub enum AlgoType {
    /// Time-weighted Average Price.
    TWAP,
    /// Volume-weighted Average Price.
    VWAP,
    /// Percentage of Volume.
    POV,
    /// Implementation Shortfall.
    IS,
}

#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[ts(export)]
/// Parameters for creating an algorithmic order.
pub struct AlgoParams {
    /// Total quantity to execute.
    pub quantity: f64,
    /// Side of the market (Bid/Ask).
    pub side: Side,
    /// Optional duration in steps.
    pub duration_steps: Option<u64>,
    /// Optional urgency parameter (0.0 to 1.0).
    pub urgency: Option<f64>,
    /// Optional participation rate (0.0 to 1.0).
    pub participation_rate: Option<f64>,
}

use crate::execution::is::IsState;
use crate::execution::pov::PovState;
use crate::execution::twap::TwapState;
use crate::execution::vwap::VwapState;
use crate::simulation::orderbook::{OrderBook, Side, Trade};

/// Internal state of an active algorithmic order.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlgoOrder {
    /// TWAP execution state.
    TWAP(TwapState),
    /// VWAP execution state.
    VWAP(VwapState),
    /// POV execution state.
    POV(PovState),
    /// IS execution state.
    IS(IsState),
}

/// Manager for handling multiple concurrent algorithmic orders.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AlgoManager {
    /// List of currently active algorithmic orders.
    pub active_orders: Vec<AlgoOrder>,
}

impl AlgoManager {
    /// Submit a new algorithmic execution order.
    pub fn submit(&mut self, algo_type: AlgoType, params: AlgoParams, start_step: u64) {
        let order = match algo_type {
            AlgoType::TWAP => AlgoOrder::TWAP(TwapState::new(
                params.quantity,
                params.side,
                start_step,
                params.duration_steps.unwrap_or(100),
            )),
            AlgoType::VWAP => AlgoOrder::VWAP(VwapState::new(
                params.quantity,
                params.side,
                start_step,
                params.duration_steps.unwrap_or(100),
                None, // Dynamic profile support can be added later
            )),
            AlgoType::POV => AlgoOrder::POV(PovState::new(
                params.quantity,
                params.side,
                params.participation_rate.unwrap_or(0.1),
            )),
            AlgoType::IS => AlgoOrder::IS(IsState::new(
                params.quantity,
                params.side,
                start_step,
                params.duration_steps.unwrap_or(100),
                params.urgency.unwrap_or(0.5),
            )),
        };
        self.active_orders.push(order);
    }

    /// Execute one simulation step for all active algorithmic orders.
    pub fn step(
        &mut self,
        current_step: u64,
        orderbook: &mut OrderBook,
        market_volume: f64,
    ) -> Vec<Trade> {
        let mut executed_trades = Vec::new();

        for algo in &mut self.active_orders {
            let trades = match algo {
                AlgoOrder::TWAP(state) => state.step(current_step, orderbook),
                AlgoOrder::VWAP(state) => state.step(current_step, orderbook),
                AlgoOrder::POV(state) => state.step(orderbook, market_volume),
                AlgoOrder::IS(state) => state.step(current_step, orderbook),
            };
            executed_trades.extend(trades);
        }

        // Remove finished orders
        self.active_orders.retain(|algo| match algo {
            AlgoOrder::TWAP(state) => !state.is_finished(current_step),
            AlgoOrder::VWAP(state) => !state.is_finished(current_step),
            AlgoOrder::POV(state) => !state.is_finished(),
            AlgoOrder::IS(state) => !state.is_finished(current_step),
        });

        executed_trades
    }
}
