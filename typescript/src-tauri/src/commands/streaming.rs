/*!
 * WebSocket price streaming commands.
 */

use crate::state::ArenaState;
use futures_util::StreamExt;
use nglab::web::polymarket::PolymarketScraper;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use tauri::{Emitter, Manager};
use tokio_tungstenite::{connect_async, tungstenite::protocol::Message};

/**
 * Real-time price update received from the Polymarket WebSocket.
 */
#[derive(serde::Serialize, Clone, Debug)]
pub struct PolymarketPriceUpdate {
    /** The unique identifier for the asset/outcome token. */
    pub asset_id: String,
    /** The latest mid-price calculated from the WebSocket message. */
    pub price: f64,
}

/**
 * Connects to the Polymarket WebSocket and starts streaming real-time prices.
 *
 * This command resolves the target market, connects to the Polymarket CLOB
 * WebSocket API, and spawns a background task to process incoming messages.
 * Updates are emitted via the `polymarket-price-update` event.
 *
 * @param app Tauri application handle for management and event emission.
 * @param market_source The Polymarket market identifier to stream.
 */
#[tauri::command]
pub async fn stream_polymarket_prices(
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
        Arc::new(std::sync::atomic::AtomicBool::new(false))
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

                use futures_util::SinkExt;
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
 *
 * Sends a stop signal to the background WebSocket task via the `ws_running` flag.
 *
 * @param app Tauri application handle to access shared state.
 */
#[tauri::command]
pub fn stop_polymarket_stream(app: tauri::AppHandle) -> Result<(), String> {
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
