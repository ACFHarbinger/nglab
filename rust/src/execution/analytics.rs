/*!
 * Execution Analytics
 *
 * Provides metrics for analyzing execution quality, including slippage,
 * market impact, and cost analysis.
 */

use crate::simulation::orderbook::Side;
use serde::{Deserialize, Serialize};
use ts_rs::TS;

#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[ts(export)]
/// Metrics regarding a single execution sequence (parent order).
pub struct ExecutionReport {
    /// Total quantity executed.
    pub total_executed: f64,
    /// Volume-Weighted Average Price of execution.
    pub vwap: f64,
    /// Benchmark price at arrival (start of order).
    pub arrival_price: f64,
    /// Total execution cost (Signed: + for Buy, - for Sell).
    pub implementation_shortfall: f64,
    /// Slippage in basis points relative to arrival price.
    pub slippage_bps: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeRecord {
    pub price: f64,
    pub quantity: f64,
    pub step: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyticsCalculator {
    pub arrival_price: f64,
    pub side: Side,
    pub trades: Vec<TradeRecord>,
}

impl AnalyticsCalculator {
    pub fn new(arrival_price: f64, side: Side) -> Self {
        Self {
            arrival_price,
            side,
            trades: Vec::new(),
        }
    }

    pub fn add_trade(&mut self, price: f64, quantity: f64, step: u64) {
        self.trades.push(TradeRecord {
            price,
            quantity,
            step,
        });
    }

    pub fn generate_report(&self) -> ExecutionReport {
        let total_qty: f64 = self.trades.iter().map(|t| t.quantity).sum();

        if total_qty == 0.0 {
            return ExecutionReport {
                total_executed: 0.0,
                vwap: 0.0,
                arrival_price: self.arrival_price,
                implementation_shortfall: 0.0,
                slippage_bps: 0.0,
            };
        }

        let total_value: f64 = self.trades.iter().map(|t| t.price * t.quantity).sum();
        let vwap = total_value / total_qty;

        let is_per_unit = match self.side {
            Side::Bid => vwap - self.arrival_price,
            Side::Ask => self.arrival_price - vwap,
        };

        let total_is = is_per_unit * total_qty;

        // Slippage in BPS
        // (IS per unit / Arrival) * 10000
        let slippage_bps = (is_per_unit / self.arrival_price) * 10000.0;

        ExecutionReport {
            total_executed: total_qty,
            vwap,
            arrival_price: self.arrival_price,
            implementation_shortfall: total_is,
            slippage_bps,
        }
    }
}
