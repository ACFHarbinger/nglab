use crate::web::polymarket::PolymarketScraper;
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio_tungstenite::{connect_async, tungstenite::protocol::Message};

/// Real-time price update received from the Polymarket WebSocket.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PolymarketPriceUpdate {
    /// The unique identifier for the asset/outcome token.
    pub asset_id: String,
    /// The latest mid-price calculated from the WebSocket message.
    pub price: f64,
}

/// Resolves a Polymarket market source (URL or slug) into outcome token IDs.
pub async fn resolve_polymarket_token_ids(market_source: &str) -> Result<Vec<String>, String> {
    // Handle Test/Mock cases
    if market_source == "1" || market_source == "test" {
        // Use a stable, high-volume market for testing (Trump 2024)
        // This ensures the test button actually streams real data.
        let target_slug = "will-trump-nominate-kevin-warsh-as-the-next-fed-chair";
        let mut scraper = PolymarketScraper::new();
        scraper
            .resolve_market(target_slug)
            .await
            .map_err(|e| format!("Failed to resolve market: {}", e))?;

        let metadata = scraper.get_metadata();
        Ok(metadata.outcomes.into_iter().map(|o| o.id).collect())
    } else {
        let mut scraper = PolymarketScraper::new();
        let mut target_id = market_source.to_string();

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
            .await
            .map_err(|e| format!("Failed to resolve market: {}", e))?;

        let metadata = scraper.get_metadata();
        Ok(metadata.outcomes.into_iter().map(|o| o.id).collect())
    }
}

/// Runs the background loop for streaming prices from Polymarket.
pub async fn stream_polymarket_prices_loop<F>(
    token_ids: Vec<String>,
    ws_running: Arc<AtomicBool>,
    on_update: F,
) where
    F: Fn(PolymarketPriceUpdate) + Send + Sync + 'static,
{
    let url = "wss://ws-subscriptions-clob.polymarket.com/ws/market";

    match connect_async(url).await {
        Ok((ws_stream, _)) => {
            eprintln!("✅ Connected to Polymarket WebSocket");
            let (mut write, mut read) = ws_stream.split();
            let (tx, mut rx) = tokio::sync::mpsc::channel::<Message>(32);
            let send_running = ws_running.clone();

            // Sender Task
            tokio::spawn(async move {
                while send_running.load(Ordering::SeqCst) {
                    if let Some(msg) = rx.recv().await {
                        if let Err(e) = write.send(msg).await {
                            eprintln!("❌ WS Send Error: {}", e);
                            break;
                        }
                    }
                }
                send_running.store(false, Ordering::SeqCst);
            });

            // 1. Subscribe
            let subscribe_msg = serde_json::json!({
                "type": "market",
                "assets_ids": token_ids,
            });
            eprintln!("📡 Sending subscription: {}", subscribe_msg);
            let _ = tx
                .send(Message::Text(subscribe_msg.to_string().into()))
                .await;

            // 2. Latency/PING task
            let ping_tx = tx.clone();
            let ping_running = ws_running.clone();
            tokio::spawn(async move {
                while ping_running.load(Ordering::SeqCst) {
                    tokio::time::sleep(tokio::time::Duration::from_secs(10)).await;
                    if (ping_tx.send(Message::Text("PING".into())).await).is_err() {
                        break;
                    }
                }
            });

            // 3. Listen loop
            while ws_running.load(Ordering::SeqCst) {
                if let Some(Ok(msg)) = read.next().await {
                    match msg {
                        Message::Text(text) => {
                            if text != "PONG" {
                                eprintln!("📨 Received WS message: {}", text);
                            }

                            if let Ok(value) = serde_json::from_str::<serde_json::Value>(&text) {
                                let mut handled = false;

                                // Format 0: Official Polymarket price_change event
                                if let Some("price_change") =
                                    value.get("event_type").and_then(|v| v.as_str())
                                {
                                    if let Some(price_changes) =
                                        value.get("price_changes").and_then(|v| v.as_array())
                                    {
                                        for change in price_changes {
                                            let bid = parse_price_value(change.get("best_bid"));
                                            let ask = parse_price_value(change.get("best_ask"));
                                            let price_raw = parse_price_value(change.get("price"));

                                            let price = match (bid, ask) {
                                                (Some(b), Some(a)) => Some((b + a) / 2.0),
                                                _ => price_raw,
                                            };

                                            if let (Some(p), Some(asset_id)) = (
                                                price,
                                                change.get("asset_id").and_then(|v| v.as_str()),
                                            ) {
                                                on_update(PolymarketPriceUpdate {
                                                    asset_id: asset_id.to_string(),
                                                    price: p,
                                                });
                                                handled = true;
                                            }
                                        }
                                    }
                                }

                                // Fallback Format 1: Generic array/object
                                if !handled {
                                    if let Some(arr) = value.as_array() {
                                        for item in arr {
                                            if let (Some(asset_id), Some(price)) = (
                                                item.get("asset_id").and_then(|v| v.as_str()),
                                                parse_price_value(item.get("price")),
                                            ) {
                                                on_update(PolymarketPriceUpdate {
                                                    asset_id: asset_id.to_string(),
                                                    price,
                                                });
                                                handled = true;
                                            }
                                        }
                                    } else if let (Some(asset_id), Some(price)) = (
                                        value.get("asset_id").and_then(|v| v.as_str()),
                                        parse_price_value(value.get("price")),
                                    ) {
                                        on_update(PolymarketPriceUpdate {
                                            asset_id: asset_id.to_string(),
                                            price,
                                        });
                                        handled = true;
                                    }
                                }

                                if !handled && text != "[]" && text != "PONG" {
                                    eprintln!("⚠️ Unhandled message format: {}", text);
                                }
                            }
                        }
                        Message::Ping(data) => {
                            let _ = tx.send(Message::Pong(data)).await;
                        }
                        Message::Close(_) => break,
                        _ => {}
                    }
                } else {
                    break;
                }
            }
            ws_running.store(false, Ordering::SeqCst);
        }
        Err(e) => {
            eprintln!("❌ Failed to connect to Polymarket WS: {}", e);
            ws_running.store(false, Ordering::SeqCst);
        }
    }
}

fn parse_price_value(val: Option<&serde_json::Value>) -> Option<f64> {
    val.and_then(|v| {
        if let Some(s) = v.as_str() {
            s.parse::<f64>().ok()
        } else {
            v.as_f64()
        }
    })
}
