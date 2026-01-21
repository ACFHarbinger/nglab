/*!
 * Visualization and logging utilities using Rerun.
 *
 * Provides real-time and post-simulation data logging
 * for order books, account stats, and model parameters.
 */

use crate::simulation::orderbook::OrderBook;

#[cfg(feature = "logging")]
use rerun::archetypes::{Points2D, Scalars};
#[cfg(feature = "logging")]
use rerun::{RecordingStream, RecordingStreamBuilder};

/**
 * Logger that streams simulation events and metrics to Rerun.
 */
pub struct RerunLogger {
    #[cfg(feature = "logging")]
    rec: Option<RecordingStream>,
}

impl RerunLogger {
    /// Creates a new RerunLogger instance.
    ///
    /// # Arguments
    /// * `enabled` - Whether logging should be active.
    #[cfg(feature = "logging")]
    pub fn new(enabled: bool) -> Self {
        if !enabled {
            return Self { rec: None };
        }
        // Save to file for robust logging in headless/CI environments
        let rec = RecordingStreamBuilder::new("nglab_arena")
            .save("arena_log.rrd")
            .expect("Failed to initialize Rerun recording stream");

        eprintln!("Rerun logging initialized to arena_log.rrd");

        Self { rec: Some(rec) }
    }

    /// Creates a no-op RerunLogger instance when logging is disabled.
    #[cfg(not(feature = "logging"))]
    pub fn new(_enabled: bool) -> Self {
        Self {}
    }

    /// Logs a single simulation step's data to Rerun.
    ///
    /// # Arguments
    /// * `step` - Current step sequence number.
    /// * `orderbook` - Reference to the current order book state.
    /// * `portfolio_value` - Current total portfolio value for tracking.
    #[cfg(feature = "logging")]
    pub fn log_step(&self, step: u64, orderbook: &OrderBook, portfolio_value: f64) {
        if let Some(rec) = &self.rec {
            rec.set_time_sequence("step", step as i64);

            // Log portfolio metrics
            rec.log("portfolio/value", &Scalars::new([portfolio_value]))
                .ok();

            // Log Market Bests
            if let Some(bid) = orderbook.best_bid() {
                rec.log("market/bid", &Scalars::new([bid])).ok();
            }
            if let Some(ask) = orderbook.best_ask() {
                rec.log("market/ask", &Scalars::new([ask])).ok();
            }

            // Log Imbalance
            let imbalance = orderbook.imbalance();
            rec.log("market/imbalance", &Scalars::new([imbalance])).ok();

            // Log Order Book Depth (Top 10 levels)
            let bids = orderbook.bid_depth(10);
            let asks = orderbook.ask_depth(10);

            // Convert to 2D points (Price, Volume)
            let bid_points: Vec<[f32; 2]> =
                bids.iter().map(|(p, v)| [*p as f32, *v as f32]).collect();
            let ask_points: Vec<[f32; 2]> =
                asks.iter().map(|(p, v)| [*p as f32, *v as f32]).collect();

            if !bid_points.is_empty() {
                rec.log(
                    "market/depth/bids",
                    &Points2D::new(bid_points).with_colors([0x00FF00FF]),
                )
                .ok();
            }
            if !ask_points.is_empty() {
                rec.log(
                    "market/depth/asks",
                    &Points2D::new(ask_points).with_colors([0xFF0000FF]),
                )
                .ok();
            }
        }
    }

    /// No-op log step when logging is disabled.
    #[cfg(not(feature = "logging"))]
    pub fn log_step(&self, _step: u64, _orderbook: &OrderBook, _portfolio_value: f64) {
        // No-op when logging feature is disabled
    }
}
