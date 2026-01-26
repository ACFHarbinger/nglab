use nglab::web::exchanges::manager::ExchangeManager;
use nglab::web::exchanges::{MarketData, MarketSearchResult};
use std::sync::Mutex as StdMutex;
use tauri::State;

#[tauri::command]
pub async fn list_exchanges(
    state: State<'_, StdMutex<ExchangeManager>>,
) -> Result<Vec<String>, String> {
    let manager = state.lock().map_err(|e| e.to_string())?;
    Ok(manager.list_exchanges())
}

#[tauri::command]
pub async fn get_active_exchange(
    state: State<'_, StdMutex<ExchangeManager>>,
) -> Result<String, String> {
    let manager = state.lock().map_err(|e| e.to_string())?;
    Ok(manager.get_active_name())
}

#[tauri::command]
pub async fn set_active_exchange(
    state: State<'_, StdMutex<ExchangeManager>>,
    name: String,
) -> Result<(), String> {
    let mut manager = state.lock().map_err(|e| e.to_string())?;
    manager.set_active_exchange(&name)
}

#[tauri::command]
pub async fn search_exchange_markets(
    state: State<'_, StdMutex<ExchangeManager>>,
    query: String,
    limit: usize,
) -> Result<Vec<MarketSearchResult>, String> {
    // 1. Lock the manager to get the active exchange Arc
    let exchange = {
        let manager = state.lock().map_err(|e| e.to_string())?;
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
    state: State<'_, StdMutex<ExchangeManager>>,
    symbol: String,
) -> Result<MarketData, String> {
    // 1. Lock the manager to get the active exchange Arc
    let exchange = {
        let manager = state.lock().map_err(|e| e.to_string())?;
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
    state: State<'_, StdMutex<ExchangeManager>>,
    app_handle: tauri::AppHandle,
    symbols: Vec<String>,
) -> Result<(), String> {
    use tauri::Emitter;

    let exchange = {
        let manager = state.lock().map_err(|e| e.to_string())?;
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
