//! Tauri backend logic for the NGLab dashboard.
//!
//! This module coordinates the interaction between the Rust simulation
//! arena and the React frontend via Tauri commands and events.

use futures_util::{SinkExt, StreamExt};
use nglab::simulation::gym::TradingEnv;
use nglab::simulation::orderbook::OrderBook;
use nglab::web::polymarket::{Frequency, PolymarketScraper};
use nglab::web::scraper::WebScraper;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use tauri::{Emitter, Manager, State};
use tokio::time::{sleep, Duration};
use tokio_tungstenite::{connect_async, tungstenite::protocol::Message};

/** Shared application state for the Tauri backend. */
struct ArenaState {
    /** Mutex-protected trading environment */
    env: Mutex<TradingEnv>,
    /** Simulation run state */
    running: Mutex<bool>,
    /** WebSocket streaming control flag */
    ws_running: Arc<AtomicBool>,
}

/** Real-time update event emitted to the frontend. */
#[derive(serde::Serialize, Clone)]
struct ArenaUpdate {
    /** Current simulation step */
    step: u64,
    /** Current mid-price from the orderbook */
    price: f64,
    /** Total portfolio value in USDC */
    portfolio_value: f64,
    /** Snapshot of the current orderbook */
    orderbook: OrderBook,
}

/**
 * Command to start the simulation loop in the background.
 */
#[tauri::command]
fn start_simulation(state: State<ArenaState>, app: tauri::AppHandle) {
    let mut running = state.running.lock().unwrap();
    if *running {
        return;
    }
    *running = true;

    tauri::async_runtime::spawn(async move {
        let state = app.state::<ArenaState>();
        loop {
            // Check if we should keep running
            {
                let running = state.running.lock().unwrap();
                if !*running {
                    break;
                }
            }

            // Perform simulation step
            let update = {
                let mut env = state.env.lock().unwrap();
                // 0 = Hold action
                let (_, _, _, _, step_info) = env.step_rs(0);

                let orderbook = env.orderbook().clone();
                let price = orderbook.mid_price().unwrap_or(0.0);

                ArenaUpdate {
                    step: step_info.total_steps,
                    price,
                    portfolio_value: step_info.portfolio_value,
                    orderbook,
                }
            };

            // Emit event to frontend
            let _ = app.emit("arena-update", &update);

            sleep(Duration::from_millis(100)).await;
        }
    });
}

/**
 * Command to stop the active simulation loop.
 */
#[tauri::command]
fn stop_simulation(state: State<ArenaState>) {
    let mut running = state.running.lock().unwrap();
    *running = false;
}

/**
 * Command to scrape Polymarket data for multiple tokens.
 */
#[tauri::command]
async fn scrape_polymarket(
    market_source: Option<String>,
    token_ids: Vec<String>,
    frequency: String,
    start_date: Option<String>,
    end_date: Option<String>,
    output_path: String,
) -> Result<(), String> {
    let freq = match frequency.as_str() {
        "1m" | "Minutely" => Frequency::Minutely,
        "1h" | "Hourly" => Frequency::Hourly,
        "1d" | "Daily" => Frequency::Daily,
        "1w" | "Weekly" => Frequency::Weekly,
        "30d" | "Monthly" => Frequency::Monthly,
        _ => return Err(format!("Invalid frequency: {}", frequency)),
    };

    let date_range = if let Some(start) = start_date {
        let start_dt = chrono::DateTime::parse_from_rfc3339(&start)
            .map_err(|e| format!("Invalid start date: {}", e))?
            .with_timezone(&chrono::Utc);
        let end_dt = if let Some(end) = end_date {
            chrono::DateTime::parse_from_rfc3339(&end)
                .map_err(|e| format!("Invalid end date: {}", e))?
                .with_timezone(&chrono::Utc)
        } else {
            chrono::Utc::now()
        };
        Some((start_dt, end_dt))
    } else {
        None
    };

    tauri::async_runtime::spawn_blocking(move || {
        let mut scraper = PolymarketScraper::new().with_frequency(freq);

        if let Some((start, end)) = date_range {
            scraper = scraper.with_date_range(start, end);
        }

        if let Some(mut source) = market_source {
            if !source.trim().is_empty() {
                // If source passes, resolve full market first to get Names

                // Clean URL if needed
                if source.starts_with("http") {
                    if let Ok(url) = url::Url::parse(&source) {
                        if let Some(segments) = url.path_segments() {
                            let segments_vec: Vec<&str> = segments.collect();
                            if let Some(last) = segments_vec.last() {
                                if !last.is_empty() {
                                    source = last.to_string();
                                } else if segments_vec.len() > 1 {
                                    source = segments_vec[segments_vec.len() - 2].to_string();
                                }
                            }
                        }
                    }
                }

                scraper
                    .resolve_market(&source)
                    .map_err(|e| format!("Failed to resolve market source: {}", e))?;

                // If user selected specific tokens, filter them now
                if !token_ids.is_empty() {
                    scraper = scraper.filter_options(token_ids);
                }
            } else {
                // Empty string source, fallback to legacy
                if !token_ids.is_empty() {
                    scraper = scraper.with_options(token_ids);
                }
            }
        } else {
            // Legacy mode: Try to infer from first token_id if it looks like a URL/Slug
            if token_ids.is_empty() {
                return Err("No input provided".to_string());
            }

            let mut target_id = token_ids[0].clone();
            let mut is_url = false;

            if target_id.starts_with("http") || !target_id.chars().all(char::is_numeric) {
                is_url = true; // Heuristic
                               // ... clean url logic ...
                if target_id.starts_with("http") {
                    if let Ok(url) = url::Url::parse(&target_id) {
                        if let Some(segments) = url.path_segments() {
                            let segments_vec: Vec<&str> = segments.collect();
                            if let Some(last) = segments_vec.last() {
                                if !last.is_empty() {
                                    target_id = last.to_string();
                                } else if segments_vec.len() > 1 {
                                    target_id = segments_vec[segments_vec.len() - 2].to_string();
                                }
                            }
                        }
                    }
                }
            }

            // Try resolve
            if scraper.resolve_market(&target_id).is_ok() {
                // Good
            } else {
                // Failure
                if !is_url {
                    // Maybe it was just a token ID?
                    scraper = scraper.with_options(token_ids);
                } else {
                    return Err(format!("Failed to resolve: {}", target_id));
                }
            }
        }

        scraper
            .download_csv(&output_path)
            .map_err(|e| format!("Scraping failed: {}", e))
    })
    .await
    .map_err(|e| format!("Task failed: {}", e))?
}

/**
 * Command to resolve a Polymarket market SLUG or URL into metadata.
 */
#[tauri::command]
async fn resolve_polymarket_id(
    input: String,
) -> Result<nglab::web::polymarket::MarketMetadata, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let mut scraper = PolymarketScraper::new();

        // precise URL parsing to get the slug (Reuse logic)
        let mut target_id = input.clone();
        if target_id.starts_with("http") {
            if let Ok(url) = url::Url::parse(&target_id) {
                if let Some(segments) = url.path_segments() {
                    let segments_vec: Vec<&str> = segments.collect();
                    if let Some(last) = segments_vec.last() {
                        if !last.is_empty() {
                            target_id = last.to_string();
                        } else if segments_vec.len() > 1 {
                            target_id = segments_vec[segments_vec.len() - 2].to_string();
                        }
                    }
                }
            }
        }

        scraper
            .resolve_market(&target_id)
            .map_err(|e| format!("Failed to resolve market: {}", e))?;

        Ok(scraper.get_metadata())
    })
    .await
    .map_err(|e| format!("Task failed: {}", e))?
}

/**
 * Command to price options using the Rough Bergomi model.
 */
#[tauri::command]
async fn pricing_rbergomi(
    params: nglab::models::rough_bergomi::RBergomiParams,
) -> Result<nglab::models::rough_bergomi::RBergomiResult, String> {
    tauri::async_runtime::spawn_blocking(move || nglab::models::rough_bergomi::simulate(params))
        .await
        .map_err(|e| format!("Simulation task failed: {}", e))?
}

/**
 * Command to price options using the Black-Scholes model.
 */
#[tauri::command]
async fn pricing_black_scholes(
    params: nglab::models::black_scholes::BlackScholesParams,
) -> Result<nglab::models::black_scholes::BlackScholesResult, String> {
    tauri::async_runtime::spawn_blocking(move || nglab::models::black_scholes::price(params))
        .await
        .map_err(|e| format!("Pricing task failed: {}", e))
}

/**
 * Command to calculate Credit Valuation Adjustment (CVA).
 */
#[tauri::command]
async fn pricing_credit_risk(
    params: nglab::models::credit_risk::CreditRiskParams,
) -> Result<nglab::models::credit_risk::CreditRiskResult, String> {
    tauri::async_runtime::spawn_blocking(move || nglab::models::credit_risk::price(params))
        .await
        .map_err(|e| format!("Pricing task failed: {}", e))?
}

/**
 * Command to price options using the Rough Heston model.
 */
#[tauri::command]
async fn pricing_rough_heston(
    params: nglab::models::rough_heston::RoughHestonParams,
) -> Result<nglab::models::rough_heston::RoughHestonResult, String> {
    tauri::async_runtime::spawn_blocking(move || nglab::models::rough_heston::simulate(params))
        .await
        .map_err(|e| format!("Simulation task failed: {}", e))?
}

/**
 * Command to predict future prices using ARIMA.
 */
#[tauri::command]
async fn predict_arima(
    params: nglab::moon::arima::ArimaParams,
) -> Result<nglab::moon::arima::ArimaResult, String> {
    tauri::async_runtime::spawn_blocking(move || nglab::moon::arima::simulate(params))
        .await
        .map_err(|e| format!("Simulation task failed: {}", e))?
}

/**
 * Command to predict future prices using GARCH.
 */
#[tauri::command]
async fn predict_garch(
    params: nglab::moon::garch::GarchParams,
) -> Result<nglab::moon::garch::GarchResult, String> {
    tauri::async_runtime::spawn_blocking(move || nglab::moon::garch::simulate(params))
        .await
        .map_err(|e| format!("Simulation task failed: {}", e))?
}

/**
 * Command to predict future prices using Holt-Winters (Exponential Smoothing).
 */
#[tauri::command]
async fn predict_holt_winters(
    params: nglab::moon::es::HoltWintersParams,
) -> Result<nglab::moon::es::HoltWintersResult, String> {
    tauri::async_runtime::spawn_blocking(move || nglab::moon::es::simulate(params))
        .await
        .map_err(|e| format!("Simulation task failed: {}", e))?
}

/**
 * Command to predict future prices using the Prophet model.
 */
// use imports moved to top

#[derive(serde::Serialize, Clone, Debug)]
struct PolymarketPriceUpdate {
    asset_id: String,
    price: f64,
}

#[tauri::command]
async fn stream_polymarket_prices(
    app: tauri::AppHandle,
    market_source: String,
) -> Result<(), String> {
    eprintln!("========================================");
    eprintln!("stream_polymarket_prices CALLED");
    eprintln!("Market source: {}", market_source);
    eprintln!("========================================");

    // 1. Resolve Token IDs (Reusing logic from scrape_polymarket roughly, but specialized)
    eprintln!("Step 1: Resolving token IDs...");
    let token_ids = tauri::async_runtime::spawn_blocking(move || {
        let mut scraper = PolymarketScraper::new();
        // Clean URL/Slug logic
        let mut target_id = market_source.clone();
        if target_id.starts_with("http") {
            if let Ok(url) = url::Url::parse(&target_id) {
                if let Some(segments) = url.path_segments() {
                    let segments_vec: Vec<&str> = segments.collect();
                    if let Some(last) = segments_vec.last() {
                        if !last.is_empty() {
                            target_id = last.to_string();
                        } else if segments_vec.len() > 1 {
                            target_id = segments_vec[segments_vec.len() - 2].to_string();
                        }
                    }
                }
            }
        }

        scraper
            .resolve_market(&target_id)
            .map_err(|e| format!("Failed to resolve market: {}", e))?;

        // Extract IDs from scraper (We might need to expose them better in Scraper or just re-get metadata)
        let metadata = scraper.get_metadata();
        Ok::<Vec<String>, String>(metadata.outcomes.into_iter().map(|o| o.id).collect())
    })
    .await
    .map_err(|e| {
        eprintln!("ERROR in spawn_blocking: {}", e);
        format!("Task failed: {}", e)
    })?
    .map_err(|e| {
        eprintln!("ERROR resolving market: {}", e);
        e
    })?;

    eprintln!("Step 1 complete: Found {} token IDs", token_ids.len());

    if token_ids.is_empty() {
        eprintln!("ERROR: No token IDs found for this market");
        return Err("No token IDs found for this market".to_string());
    }

    eprintln!(
        "🚀 Starting WebSocket stream for {} token IDs",
        token_ids.len()
    );
    eprintln!("Token IDs: {:?}", &token_ids[0..token_ids.len().min(3)]);

    // Set the streaming flag to true
    if let Some(state) = app.try_state::<ArenaState>() {
        state.ws_running.store(true, Ordering::SeqCst);
    }

    // 2. Spawn WebSocket Task (Clone app handle and running flag before moving into task)
    let app_handle = app.clone();
    let ws_running = if let Some(state) = app.try_state::<ArenaState>() {
        state.ws_running.clone()
    } else {
        Arc::new(AtomicBool::new(false))
    };

    tauri::async_runtime::spawn(async move {
        eprintln!("📡 WebSocket task spawned, connecting to Polymarket...");
        let url = "wss://ws-subscriptions-clob.polymarket.com/ws/market";
        match connect_async(url).await {
            Ok((ws_stream, _)) => {
                let (mut write, mut read) = ws_stream.split();

                // 3. Subscribe (Correct Polymarket WebSocket format)
                // See: https://docs.polymarket.com/developers/CLOB/websocket/wss-overview
                let subscribe_msg = serde_json::json!({
                    "type": "market",
                    "assets_ids": token_ids,  // Note: plural with underscore!
                });

                eprintln!("📤 Sending subscription: {}", subscribe_msg.to_string());

                if let Err(e) = write.send(Message::Text(subscribe_msg.to_string())).await {
                    eprintln!("❌ Failed to send subscription: {}", e);
                    return;
                }

                // 4. Listen Loop
                eprintln!("✓ Subscription sent");
                eprintln!(
                    "📡 WebSocket connected successfully. Subscribed to {} assets",
                    token_ids.len()
                );

                // TEST: Emit a test event to verify event propagation
                let test_update = PolymarketPriceUpdate {
                    asset_id: "TEST".to_string(),
                    price: 0.99,
                };
                eprintln!("🧪 Emitting test event...");
                if let Err(e) = app_handle.emit("polymarket-price-update", &test_update) {
                    eprintln!("  ❌ TEST EVENT FAILED TO EMIT: {}", e);
                } else {
                    eprintln!("  ✅ Test event emitted - check browser console!");
                }

                eprintln!("⏳ Waiting for messages from Polymarket...");

                let mut msg_count = 0;
                while ws_running.load(Ordering::SeqCst) {
                    if let Some(msg) = read.next().await {
                        msg_count += 1;
                        eprintln!(
                            "📨 Message #{} received (type: {:?})",
                            msg_count,
                            match &msg {
                                Ok(Message::Text(_)) => "Text",
                                Ok(Message::Binary(_)) => "Binary",
                                Ok(Message::Ping(_)) => "Ping",
                                Ok(Message::Pong(_)) => "Pong",
                                Ok(Message::Close(_)) => "Close",
                                Ok(Message::Frame(_)) => "Frame",
                                Err(_) => "Error",
                            }
                        );

                        match msg {
                            Ok(Message::Text(text)) => {
                                eprintln!("📄 WS RECV TEXT: {}", text);

                                // Parse JSON response
                                if let Ok(value) = serde_json::from_str::<serde_json::Value>(&text)
                                {
                                    // Try multiple Polymarket API formats
                                    let mut updated = false;

                                    // Format 0: Official Polymarket price_change event (Sept 2025)
                                    // {"event_type": "price_change", "price_changes": [{"asset_id": "...", "best_bid": "0.52", "best_ask": "0.53"}]}
                                    if let Some(event_type) =
                                        value.get("event_type").and_then(|s| s.as_str())
                                    {
                                        if event_type == "price_change" {
                                            if let Some(price_changes) = value
                                                .get("price_changes")
                                                .and_then(|v| v.as_array())
                                            {
                                                for change in price_changes {
                                                    if let Some(asset_id) = change
                                                        .get("asset_id")
                                                        .and_then(|s| s.as_str())
                                                    {
                                                        // Use midpoint of best_bid and best_ask
                                                        let bid_opt =
                                                            change.get("best_bid").and_then(|b| {
                                                                if let Some(s) = b.as_str() {
                                                                    s.parse::<f64>().ok()
                                                                } else {
                                                                    b.as_f64()
                                                                }
                                                            });
                                                        let ask_opt =
                                                            change.get("best_ask").and_then(|a| {
                                                                if let Some(s) = a.as_str() {
                                                                    s.parse::<f64>().ok()
                                                                } else {
                                                                    a.as_f64()
                                                                }
                                                            });

                                                        if let (Some(bid), Some(ask)) =
                                                            (bid_opt, ask_opt)
                                                        {
                                                            let price = (bid + ask) / 2.0; // Midpoint price
                                                            eprintln!("✓ Emitting price_change: {} = ${} (bid:{}, ask:{})",
                                                            asset_id, price, bid, ask);
                                                            let update = PolymarketPriceUpdate {
                                                                asset_id: asset_id.to_string(),
                                                                price,
                                                            };

                                                            // Try emitting to all windows
                                                            // Emit event to frontend
                                                            if let Err(e) = app_handle.emit(
                                                                "polymarket-price-update",
                                                                &update,
                                                            ) {
                                                                eprintln!(
                                                                    "  ❌ Failed to emit event: {}",
                                                                    e
                                                                );
                                                            } else {
                                                                eprintln!("  ✅ Event emitted successfully");
                                                            }

                                                            updated = true;
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }

                                    // Format 1: Direct array of updates
                                    if !updated {
                                        if let Some(arr) = value.as_array() {
                                            for item in arr {
                                                if let Some(asset_id) =
                                                    item.get("asset_id").and_then(|s| s.as_str())
                                                {
                                                    // Try getting price as string or number
                                                    let price_opt =
                                                        item.get("price").and_then(|p| {
                                                            if let Some(s) = p.as_str() {
                                                                s.parse::<f64>().ok()
                                                            } else {
                                                                p.as_f64()
                                                            }
                                                        });

                                                    if let Some(price) = price_opt {
                                                        eprintln!(
                                                            "✓ Emitting price update: {} = ${}",
                                                            asset_id, price
                                                        );
                                                        let update = PolymarketPriceUpdate {
                                                            asset_id: asset_id.to_string(),
                                                            price,
                                                        };
                                                        // Emit event to frontend
                                                        if let Err(e) = app_handle.emit(
                                                            "polymarket-price-update",
                                                            &update,
                                                        ) {
                                                            eprintln!("  ❌ Failed to emit: {}", e);
                                                        } else {
                                                            eprintln!("  ✅ Emitted");
                                                        }
                                                        updated = true;
                                                    }
                                                }
                                            }
                                        }
                                    }

                                    // Format 2: Single object with asset_id and price
                                    if !updated {
                                        if let Some(asset_id) =
                                            value.get("asset_id").and_then(|s| s.as_str())
                                        {
                                            let price_opt = value.get("price").and_then(|p| {
                                                if let Some(s) = p.as_str() {
                                                    s.parse::<f64>().ok()
                                                } else {
                                                    p.as_f64()
                                                }
                                            });

                                            if let Some(price) = price_opt {
                                                eprintln!(
                                                    "✓ Emitting single price update: {} = ${}",
                                                    asset_id, price
                                                );
                                                let update = PolymarketPriceUpdate {
                                                    asset_id: asset_id.to_string(),
                                                    price,
                                                };
                                                let _ = app_handle
                                                    .emit("polymarket-price-update", &update);
                                                updated = true;
                                            }
                                        }
                                    }

                                    // Format 3: Nested data structure (common in some APIs)
                                    if !updated {
                                        if let Some(data) = value.get("data") {
                                            if let Some(arr) = data.as_array() {
                                                for item in arr {
                                                    if let Some(asset_id) = item
                                                        .get("asset_id")
                                                        .and_then(|s| s.as_str())
                                                    {
                                                        let price_opt =
                                                            item.get("price").and_then(|p| {
                                                                if let Some(s) = p.as_str() {
                                                                    s.parse::<f64>().ok()
                                                                } else {
                                                                    p.as_f64()
                                                                }
                                                            });

                                                        if let Some(price) = price_opt {
                                                            eprintln!("✓ Emitting nested price update: {} = ${}", asset_id, price);
                                                            let update = PolymarketPriceUpdate {
                                                                asset_id: asset_id.to_string(),
                                                                price,
                                                            };
                                                            // Emit event to frontend
                                                            if let Err(e) = app_handle.emit(
                                                                "polymarket-price-update",
                                                                &update,
                                                            ) {
                                                                eprintln!(
                                                                    "  ❌ Failed to emit: {}",
                                                                    e
                                                                );
                                                            } else {
                                                                eprintln!("  ✅ Emitted");
                                                            }
                                                            updated = true;
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }

                                    if !updated {
                                        eprintln!(
                                            "⚠ Could not parse price update from message format"
                                        );
                                    }
                                } else {
                                    eprintln!("⚠ Failed to parse JSON from WebSocket message");
                                }
                            }
                            Ok(Message::Ping(data)) => {
                                eprintln!("🏓 Received Ping, sending Pong");
                                let _ = write.send(Message::Pong(data)).await;
                            }
                            Ok(Message::Pong(_)) => {
                                eprintln!("🏓 Received Pong");
                            }
                            Ok(Message::Close(frame)) => {
                                eprintln!("❌ WebSocket connection closed: {:?}", frame);
                                break;
                            }
                            Ok(Message::Binary(data)) => {
                                eprintln!("📦 Received binary data: {} bytes", data.len());
                            }
                            Err(e) => {
                                eprintln!("❌ WebSocket Error: {}", e);
                                break;
                            }
                            _ => {
                                eprintln!("❓ Unknown message type");
                            }
                        }
                    } else {
                        // No more messages, exit loop
                        break;
                    }
                }
                eprintln!("📡 WebSocket stream stopped by user or connection closed");
                ws_running.store(false, Ordering::SeqCst);
            }
            Err(e) => {
                eprintln!("Failed to connect to Polymarket WS: {}", e);
                ws_running.store(false, Ordering::SeqCst);
            }
        }
    });

    eprintln!("stream_polymarket_prices returning Ok - WebSocket task spawned");
    eprintln!("========================================");
    Ok(())
}

/**
 * Stops the active Polymarket WebSocket price stream.
 */
#[tauri::command]
fn stop_polymarket_stream(app: tauri::AppHandle) -> Result<(), String> {
    eprintln!("🛑 Stopping Polymarket WebSocket stream...");

    if let Some(state) = app.try_state::<ArenaState>() {
        let was_running = state.ws_running.swap(false, Ordering::SeqCst);
        if was_running {
            eprintln!("✓ WebSocket stream stop signal sent");
            Ok(())
        } else {
            eprintln!("⚠ No active WebSocket stream found");
            Err("No active stream to stop".to_string())
        }
    } else {
        Err("Failed to access stream state".to_string())
    }
}

/**
 * Command to list available trained models in the python/trained_models directory.
 */
#[tauri::command]
async fn list_trained_models(app: tauri::AppHandle) -> Result<Vec<String>, String> {
    // Resolve path relative to app? Or absolute?
    // Assuming simplified relative structure for dev, but in prod this needs resource path.
    // For now, let's look in the user's workspace/python/trained_models
    // We can assume the CWD is the project root in dev mode.
    // In dev: tauri dev runs from project root usually?
    // The user says "python/trained_models" exists.

    let path = std::path::Path::new("../../python/trained_models");
    if !path.exists() {
        return Ok(vec![]);
    }

    let mut models = Vec::new();
    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries {
            if let Ok(entry) = entry {
                let path = entry.path();
                if path.extension().map_or(false, |ext| ext == "pt") {
                    if let Some(name) = path.file_stem().and_then(|s| s.to_str()) {
                        models.push(name.to_string());
                    }
                }
            }
        }
    }
    models.sort();
    Ok(models)
}

/**
 * Command to run inference on a selected trained model.
 */
#[derive(serde::Serialize)]
struct PredictionResponse {
    status: String,
    prediction: Option<Vec<f64>>, // Flexible, maybe scalar or array
    message: Option<String>,
}

#[derive(serde::Deserialize)]
struct InferenceOutput {
    status: String,
    prediction: Option<serde_json::Value>, // Can be number list or nested
    message: Option<String>,
    // metadata: ...
}

#[tauri::command]
async fn predict_trained_model(
    model_name: String,
    input_data: Vec<f64>,
) -> Result<Vec<f64>, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let model_path = format!("../../python/trained_models/{}.pt", model_name);

        let mut cmd = std::process::Command::new("python3");
        cmd.arg("../../python/src/infer.py")
            .arg("--model_path")
            .arg(&model_path)
            .arg("--input_json")
            .arg(serde_json::to_string(&input_data).unwrap_or("[]".to_string()));

        let output = cmd
            .output()
            .map_err(|e| format!("Failed to execute python: {}", e))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!("Inference script failed: {}", stderr));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let result: InferenceOutput = serde_json::from_str(&stdout)
            .map_err(|e| format!("Failed to parse inference output: {} | raw: {}", e, stdout))?;

        if result.status == "success" {
            // Normalize prediction to Vec<f64>
            if let Some(val) = result.prediction {
                if let Some(arr) = val.as_array() {
                    let vec: Vec<f64> = arr.iter().filter_map(|v| v.as_f64()).collect();
                    return Ok(vec);
                } else if let Some(num) = val.as_f64() {
                    return Ok(vec![num]);
                }
            }
            Err("Empty or invalid prediction format".to_string())
        } else {
            Err(result.message.unwrap_or("Unknown error".to_string()))
        }
    })
    .await
    .map_err(|e| format!("Task failed: {}", e))?
}

#[tauri::command]
async fn predict_prophet(
    params: nglab::moon::prophet::ProphetParams,
) -> Result<nglab::moon::prophet::ProphetResult, String> {
    tauri::async_runtime::spawn_blocking(move || nglab::moon::prophet::simulate(params))
        .await
        .map_err(|e| format!("Simulation task failed: {}", e))?
}

/**
 * Entry point for the Tauri application.
 */
#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    // Initialize TradingEnv with default parameters
    let env = TradingEnv::new(10000.0, 0.001, 30, 1000, false);

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
        .invoke_handler(tauri::generate_handler![
            start_simulation,
            stop_simulation,
            scrape_polymarket,
            stream_polymarket_prices,
            stop_polymarket_stream,
            resolve_polymarket_id,
            pricing_rbergomi,
            pricing_black_scholes,
            pricing_credit_risk,
            pricing_rough_heston,
            predict_arima,
            predict_garch,
            predict_holt_winters,
            predict_prophet,
            list_trained_models,
            predict_trained_model
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
