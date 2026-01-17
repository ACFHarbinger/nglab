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
    /** Complete snapshot of the order book for visualization. */
    pub orderbook: OrderBook,
}
