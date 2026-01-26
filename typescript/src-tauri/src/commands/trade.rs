/*!
 * Trade submission commands.
 */

use crate::state::{ArenaState, PaperState};
use nglab::simulation::orderbook::{Order, PegReference, Side};
use std::sync::atomic::Ordering;
use tauri::State;

/**
 * Submit a Limit order.
 */
#[tauri::command]
pub fn submit_limit_order(
    arena: State<ArenaState>,
    paper: State<PaperState>,
    price: f64,
    quantity: f64,
    side: Side,
) -> Option<u64> {
    if paper.active.load(Ordering::SeqCst) {
        let mut account = paper.account.lock().unwrap();
        let timestamp = arena.env.lock().unwrap().info().total_steps;
        let order = Order::new(
            0,
            price,
            quantity,
            side,
            nglab::simulation::orderbook::OrderType::Limit,
            timestamp,
        );
        Some(account.submit_order(order))
    } else {
        let mut env = arena.env.lock().unwrap();
        let orderbook = env.orderbook_mut();
        orderbook
            .submit_limit_order(price, quantity, side)
            .ok()
            .map(|(id, _)| id)
    }
}

/**
 * Submit a Fill-or-Kill (FOK) order.
 */
#[tauri::command]
pub fn submit_fok_order(
    arena: State<ArenaState>,
    paper: State<PaperState>,
    price: f64,
    quantity: f64,
    side: Side,
) -> Option<u64> {
    if paper.active.load(Ordering::SeqCst) {
        let mut account = paper.account.lock().unwrap();
        let timestamp = arena.env.lock().unwrap().info().total_steps; // Use sim time
        let order = Order::new(
            0,
            price,
            quantity,
            side,
            nglab::simulation::orderbook::OrderType::FillOrKill,
            timestamp,
        );
        Some(account.submit_order(order))
    } else {
        let mut env = arena.env.lock().unwrap();
        let orderbook = env.orderbook_mut();
        orderbook
            .submit_fok_order(price, quantity, side)
            .ok()
            .map(|(id, _)| id)
    }
}

/**
 * Submit an Immediate-or-Cancel (IOC) order.
 */
#[tauri::command]
pub fn submit_ioc_order(
    arena: State<ArenaState>,
    paper: State<PaperState>,
    price: f64,
    quantity: f64,
    side: Side,
) -> Option<u64> {
    if paper.active.load(Ordering::SeqCst) {
        let mut account = paper.account.lock().unwrap();
        let timestamp = arena.env.lock().unwrap().info().total_steps;
        let order = Order::new(
            0,
            price,
            quantity,
            side,
            nglab::simulation::orderbook::OrderType::ImmediateOrCancel,
            timestamp,
        );
        Some(account.submit_order(order))
    } else {
        let mut env = arena.env.lock().unwrap();
        let orderbook = env.orderbook_mut();
        orderbook
            .submit_ioc_order(price, quantity, side)
            .ok()
            .map(|(id, _)| id)
    }
}

/**
 * Submit a Bracket order (Limit Entry + SL + TP).
 */
#[tauri::command]
pub fn submit_bracket_order(
    arena: State<ArenaState>,
    paper: State<PaperState>,
    price: f64,
    quantity: f64,
    side: Side,
    sl_price: f64,
    tp_price: f64,
) -> u64 {
    if paper.active.load(Ordering::SeqCst) {
        let mut account = paper.account.lock().unwrap();
        let timestamp = arena.env.lock().unwrap().info().total_steps;
        let mut order = Order::new(
            0,
            price,
            quantity,
            side,
            nglab::simulation::orderbook::OrderType::Limit,
            timestamp,
        );
        order.bracket_sl = Some(sl_price);
        order.bracket_tp = Some(tp_price);
        account.submit_order(order)
    } else {
        let mut env = arena.env.lock().unwrap();
        let orderbook = env.orderbook_mut();
        orderbook.submit_bracket_order(price, quantity, side, sl_price, tp_price)
    }
}

/**
 * Submit a Pegged order.
 */
#[tauri::command]
pub fn submit_pegged_order(
    arena: State<ArenaState>,
    paper: State<PaperState>,
    quantity: f64,
    side: Side,
    peg_reference: PegReference,
    peg_offset: f64,
) -> Option<u64> {
    if paper.active.load(Ordering::SeqCst) {
        let mut account = paper.account.lock().unwrap();
        let timestamp = arena.env.lock().unwrap().info().total_steps;
        let mut order = Order::new(
            0,
            0.0,
            quantity,
            side,
            nglab::simulation::orderbook::OrderType::Pegged,
            timestamp,
        );
        order.peg_reference = Some(peg_reference);
        order.peg_offset = Some(peg_offset);
        Some(account.submit_order(order))
    } else {
        let mut env = arena.env.lock().unwrap();
        let orderbook = env.orderbook_mut();
        orderbook.submit_pegged_order(quantity, side, peg_reference, peg_offset)
    }
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
