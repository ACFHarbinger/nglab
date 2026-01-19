/*!
 * Tauri backend logic for the NGLab dashboard.
 *
 * This module coordinates the interaction between the Rust simulation
 * arena and the React frontend via Tauri commands and events.
 */

pub mod commands;
pub mod state;

use crate::commands::auth::AuthState;
use crate::commands::vault::VaultState;
use crate::state::ArenaState;
use nglab::simulation::gym::TradingEnv;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};
use tauri::Manager;

/**
 * Entry point for the Tauri application library.
 *
 * Initializes the global application state, configures the Tauri builder
 * with necessary plugins and invoke handlers, and starts the application run loop.
 */
#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    // Initialize TradingEnv with default parameters and fixed seed for reproducibility
    let env = TradingEnv::new(10000.0, 0.001, 30, 1000, false, Some(42));

    let state = ArenaState {
        env: Mutex::new(env),
        running: Mutex::new(false),
        ws_running: Arc::new(AtomicBool::new(false)),
    };

    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .manage(state)
        .manage(AuthState::default())
        .manage(VaultState::default())
        .invoke_handler(tauri::generate_handler![
            commands::simulation::start_simulation,
            commands::simulation::stop_simulation,
            commands::scraping::scrape_polymarket,
            commands::streaming::stream_polymarket_prices,
            commands::streaming::stop_polymarket_stream,
            commands::scraping::resolve_polymarket_id,
            commands::pricing::pricing_rbergomi,
            commands::pricing::pricing_black_scholes,
            commands::pricing::pricing_credit_risk,
            commands::pricing::pricing_rough_heston,
            commands::moon::predict_arima,
            commands::moon::predict_garch,
            commands::moon::predict_holt_winters,
            commands::moon::predict_prophet,
            commands::inference::list_trained_models,
            commands::inference::predict_trained_model,
            commands::inference::train_model,
            commands::inference::list_csv_columns,
            commands::auth::create_account,
            commands::auth::login,
            commands::auth::logout,
            commands::auth::get_current_user,
            commands::vault::unlock_vault,
            commands::vault::lock_vault,
            commands::vault::is_vault_unlocked,
            commands::vault::add_vault_secret,
            commands::vault::list_vault_secrets,
            commands::vault::get_vault_secret,
            commands::vault::delete_vault_secret,
            commands::integrations::save_polymarket_integration,
            commands::integrations::list_integrations,
            commands::integrations::delete_integration,
            commands::integrations::get_trending_markets,
            commands::integrations::search_polymarket_markets,
            commands::integrations::get_open_polymarket_markets,
            commands::integrations::get_public_polymarket_markets,
            commands::integrations::search_public_polymarket_markets
        ])
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|app_handle, event| match event {
            tauri::RunEvent::ExitRequested { .. } => {
                if let Ok(mut running) = app_handle.state::<ArenaState>().running.lock() {
                    *running = false;
                }
            }
            _ => {}
        });
}
