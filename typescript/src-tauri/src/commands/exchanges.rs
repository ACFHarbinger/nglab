use crate::commands::vault::VaultState;
use futures_util::future::join_all;
use nglab::security::vault::VaultResponse;
use nglab::web::exchanges::manager::ExchangeManager;
use nglab::web::exchanges::{MarketData, MarketSearchResult};
use std::collections::HashMap;
use tauri::State;
use tokio::sync::Mutex as TokioMutex;

#[tauri::command]
pub async fn list_exchanges(
    state: State<'_, TokioMutex<ExchangeManager>>,
) -> Result<Vec<String>, String> {
    let manager = state.lock().await;
    Ok(manager.list_exchanges())
}

#[tauri::command]
pub async fn get_active_exchange(
    state: State<'_, TokioMutex<ExchangeManager>>,
) -> Result<String, String> {
    let manager = state.lock().await;
    Ok(manager.get_active_name())
}

#[tauri::command]
pub async fn set_active_exchange(
    state: State<'_, TokioMutex<ExchangeManager>>,
    name: String,
) -> Result<(), String> {
    let mut manager = state.lock().await;
    manager.set_active_exchange(&name)
}

#[tauri::command]
pub async fn search_exchange_markets(
    state: State<'_, TokioMutex<ExchangeManager>>,
    query: String,
    limit: usize,
) -> Result<Vec<MarketSearchResult>, String> {
    // 1. Lock the manager to get the active exchange Arc
    let exchange = {
        let manager = state.lock().await;
        manager.get_active_exchange()
    }; // manager lock dropped here

    // 2. Lock the exchange (tokio Mutex) and await the result
    let exchange_lock = exchange.lock().await;
    exchange_lock
        .search_markets(&query, limit)
        .await
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn get_exchange_market_data(
    state: State<'_, TokioMutex<ExchangeManager>>,
    symbol: String,
) -> Result<MarketData, String> {
    // 1. Lock the manager to get the active exchange Arc
    let exchange = {
        let manager = state.lock().await;
        manager.get_active_exchange()
    }; // manager lock dropped here

    // 2. Lock the exchange (tokio Mutex) and await the result
    let exchange_lock = exchange.lock().await;
    exchange_lock
        .get_market_data(&symbol)
        .await
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn stream_exchange_prices(
    state: State<'_, TokioMutex<ExchangeManager>>,
    app_handle: tauri::AppHandle,
    symbols: Vec<String>,
) -> Result<(), String> {
    use tauri::Emitter;

    let exchange = {
        let manager = state.lock().await;
        manager.get_active_exchange()
    };

    tokio::spawn(async move {
        let exchange_lock = exchange.lock().await;
        let res = exchange_lock
            .stream_prices(
                symbols,
                Box::new(move |update| {
                    let _ = app_handle.emit("price-update", update);
                }),
            )
            .await;

        if let Err(e) = res {
            eprintln!("Stream error: {}", e);
        }
    });

    Ok(())
}

#[tauri::command]
pub async fn get_all_exchange_prices(
    state: State<'_, TokioMutex<ExchangeManager>>,
    symbol: String,
) -> Result<HashMap<String, MarketData>, String> {
    let exchanges = {
        let manager = state.lock().await;
        manager.get_all_exchanges()
    };

    let mut tasks = Vec::new();
    for (name, exchange) in exchanges {
        let symbol_clone = symbol.clone();
        tasks.push(async move {
            let exchange_lock = exchange.lock().await;
            (name, exchange_lock.get_market_data(&symbol_clone).await)
        });
    }

    let raw_results = join_all(tasks).await;
    let mut results = HashMap::new();
    for (name, data_res) in raw_results {
        if let Ok(data) = data_res {
            results.insert(name, data);
        }
    }

    Ok(results)
}

#[tauri::command]
pub async fn reconnect_exchanges(
    state: State<'_, TokioMutex<ExchangeManager>>,
    vault_state: State<'_, VaultState>,
) -> Result<VaultResponse<()>, String> {
    let master_key = if let Ok(key_guard) = vault_state.master_key.lock() {
        match key_guard.as_ref() {
            Some(key) => key.clone(),
            None => return Ok(VaultResponse::error("Vault is locked")),
        }
    } else {
        return Err("Failed to lock vault state".to_string());
    };

    let manager = state.lock().await;
    match manager.connect_all_from_vault(&master_key).await {
        Ok(_) => Ok(VaultResponse::success("Exchanges reconnected", None)),
        Err(e) => Ok(VaultResponse::error(&format!("Failed to reconnect: {}", e))),
    }
}
