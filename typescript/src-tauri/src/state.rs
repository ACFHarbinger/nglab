/*!
 * Shared state and event types for the Tauri backend.
 */

use nglab::simulation::gym::TradingEnv;
use nglab::simulation::orderbook::OrderBook;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

/**
 * Shared application state for the Tauri backend.
 */
pub struct ArenaState {
    /** Mutex-protected trading environment from the core library. */
    pub env: Mutex<TradingEnv>,
    /** Boolean flag indicating if the simulation loop is currently active. */
    pub running: Mutex<bool>,
    /** Atomic flag used to control the background WebSocket price streaming. */
    pub ws_running: Arc<AtomicBool>,
    /** Global debug mode toggle. */
    pub debug_mode: Arc<AtomicBool>,
    /** Currently active machine learning model for inference. */
    pub active_model: Mutex<Option<String>>,
}

/**
 * Shared state for Paper Trading.
 */
pub struct PaperState {
    /** Mutex-protected paper account. */
    pub account: Mutex<nglab::simulation::paper_trading::PaperAccount>,
    /** Flag indicating if paper mode is active in the UI. */
    pub active: Arc<AtomicBool>,
}

/**
 * Real-time update event emitted to the frontend during simulation.
 */
#[derive(serde::Serialize, Clone)]
pub struct ArenaUpdate {
    /** Current simulation step sequence number. */
    pub step: u64,
    /** Current mid-price extracted from the order book. */
    pub price: f64,
    /** Total account portfolio value including cash and assets. */
    pub portfolio_value: f64,
    /** Current real-time risk score (0-100). */
    pub risk_score: u8,
    /** Current peak-to-trough drawdown. */
    pub current_drawdown: f64,
    /** Value at Risk (VaR). */
    pub current_var: f64,
    /** Complete snapshot of the order book for visualization. */
    pub orderbook: OrderBook,
    /** Active algorithmic execution orders. */
    pub algo_orders: Vec<nglab::execution::AlgoOrder>,
    /** Current position. */
    pub position: f64,
    /** Whether market maker mode is active. */
    pub mm_active: bool,
    /** Realized PnL from market making. */
    pub mm_realized_pnl: f64,
    pub pnl: f64,
    pub return_pct: f64,
    pub max_drawdown: f64,
    pub volatility: f64,
}
