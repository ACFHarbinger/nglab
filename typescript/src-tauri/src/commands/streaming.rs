/*!
 * WebSocket price streaming commands.
 */

use crate::state::ArenaState;
use nglab::web::streaming::{resolve_polymarket_token_ids, stream_polymarket_prices_loop};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use tauri::{Emitter, Manager};

/**
 * Connects to the Polymarket WebSocket and starts streaming real-time prices.
 *
 * This command resolves the target market, connects to the Polymarket CLOB
 * WebSocket API, and spawns a background task to process incoming messages.
 * Updates are emitted via the `polymarket-price-update` event.
 */
#[tauri::command]
pub async fn stream_polymarket_prices(
    app: tauri::AppHandle,
    market_source: String,
) -> Result<nglab::web::polymarket::MarketMetadata, String> {
    eprintln!("🚀 Starting Polymarket stream for: {}", market_source);

    // 1. Resolve Token IDs and Metadata
    let (token_ids, metadata) = resolve_polymarket_token_ids(&market_source)
        .await
        .map_err(|e| format!("Failed to resolve: {}", e))?;

    if token_ids.is_empty() {
        return Err("No token IDs found for this market".to_string());
    }

    // Set the streaming flag to true
    if let Some(state) = app.try_state::<ArenaState>() {
        state.ws_running.store(true, Ordering::SeqCst);
    }

    // 2. Spawn WebSocket Task
    let app_handle = app.clone();
    let ws_running = if let Some(state) = app.try_state::<ArenaState>() {
        state.ws_running.clone()
    } else {
        Arc::new(std::sync::atomic::AtomicBool::new(false))
    };

    tauri::async_runtime::spawn(async move {
        stream_polymarket_prices_loop(token_ids, ws_running, move |update| {
            if let Err(e) = app_handle.emit("polymarket-price-update", &update) {
                eprintln!("❌ Failed to emit price update: {}", e);
            }
        })
        .await;
    });

    Ok(metadata)
}

/**
 * Stops the active Polymarket WebSocket price stream.
 */
#[tauri::command]
pub fn stop_polymarket_stream(app: tauri::AppHandle) -> Result<(), String> {
    if let Some(state) = app.try_state::<ArenaState>() {
        if state.ws_running.swap(false, Ordering::SeqCst) {
            Ok(())
        } else {
            Err("No active stream to stop".to_string())
        }
    } else {
        Err("Failed to access stream state".to_string())
    }
}
