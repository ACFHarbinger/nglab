/*!
 * Trade submission commands.
 */

use crate::state::ArenaState;
use nglab::simulation::orderbook::{PegReference, Side};
use tauri::State;

/**
 * Submit a Fill-or-Kill (FOK) order.
 */
#[tauri::command]
pub fn submit_fok_order(
    state: State<ArenaState>,
    price: f64,
    quantity: f64,
    side: Side,
) -> Option<u64> {
    let mut env = state.env.lock().unwrap();
    let orderbook = env.orderbook_mut();
    orderbook
        .submit_fok_order(price, quantity, side)
        .ok()
        .map(|(id, _)| id)
}

/**
 * Submit an Immediate-or-Cancel (IOC) order.
 */
#[tauri::command]
pub fn submit_ioc_order(
    state: State<ArenaState>,
    price: f64,
    quantity: f64,
    side: Side,
) -> Option<u64> {
    let mut env = state.env.lock().unwrap();
    let orderbook = env.orderbook_mut();
    orderbook
        .submit_ioc_order(price, quantity, side)
        .ok()
        .map(|(id, _)| id)
}

/**
 * Submit a Bracket order (Limit Entry + SL + TP).
 */
#[tauri::command]
pub fn submit_bracket_order(
    state: State<ArenaState>,
    price: f64,
    quantity: f64,
    side: Side,
    sl_price: f64,
    tp_price: f64,
) -> u64 {
    let mut env = state.env.lock().unwrap();
    let orderbook = env.orderbook_mut();
    orderbook.submit_bracket_order(price, quantity, side, sl_price, tp_price)
}

/**
 * Submit a Pegged order.
 */
#[tauri::command]
pub fn submit_pegged_order(
    state: State<ArenaState>,
    quantity: f64,
    side: Side,
    peg_reference: PegReference,
    peg_offset: f64,
) -> Option<u64> {
    let mut env = state.env.lock().unwrap();
    let orderbook = env.orderbook_mut();
    orderbook.submit_pegged_order(quantity, side, peg_reference, peg_offset)
}

/**
 * Submit an Algorithmic Execution order (TWAP, VWAP, POV).
 */
#[tauri::command]
pub fn submit_algo_order(
    state: State<ArenaState>,
    algo_type: nglab::execution::AlgoType,
    params: nglab::execution::AlgoParams,
) {
    let mut env = state.env.lock().unwrap();
    let start_step = env.info().total_steps;
    env.algo_manager_mut().submit(algo_type, params, start_step);
}

/**
 * Start the automated Market Maker.
 */
#[tauri::command]
pub fn start_market_maker(
    state: State<ArenaState>,
    config: nglab::simulation::market_maker::MarketMakerConfig,
) {
    let mut env = state.env.lock().unwrap();
    let mm = env.market_maker_mut();
    mm.config = config;
    mm.active = true;
}

/**
 * Stop the automated Market Maker.
 */
#[tauri::command]
pub fn stop_market_maker(state: State<ArenaState>) {
    let mut env_guard = state.env.lock().unwrap();
    let env = &mut *env_guard;
    env.market_maker.active = false;
    env.market_maker.cancel_quotes(&mut env.orderbook);
}

/**
 * Submit a Multi-Leg Spread Order.
 */
#[tauri::command]
pub fn submit_spread_order(
    state: State<ArenaState>,
    order: nglab::simulation::spreads::SpreadOrder,
) {
    let mut env = state.env.lock().unwrap();
    env.submit_spread_order(order);
}
