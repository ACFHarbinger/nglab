use crate::web::polymarket::{MarketMetadata, PolymarketScraper};
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

/// Resolves a Polymarket market source (URL or slug) into outcome token IDs and metadata.
pub async fn resolve_polymarket_token_ids(
    market_source: &str,
) -> Result<(Vec<String>, MarketMetadata), String> {
    let mut scraper = PolymarketScraper::new();

    // Handle Test/Mock cases
    if market_source == "1" || market_source == "test" {
        // Use a stable, high-volume market for testing (Warsh)
        let target_slug = "will-trump-nominate-kevin-warsh-as-the-next-fed-chair";
        scraper
            .resolve_market(target_slug)
            .await
            .map_err(|e| format!("Failed to resolve market: {}", e))?;
    } else {
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
    }

    let metadata = scraper.get_metadata();
    let token_ids = metadata.outcomes.iter().map(|o| o.id.clone()).collect();
    Ok((token_ids, metadata))
}

/// Events emitted by the streaming loop.
#[derive(Serialize, Clone, Debug)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamEvent {
    /// New price update
    Data { asset_id: String, price: f64 },
    /// Periodic health/latency update
    Health { latency_ms: u64, msgs_per_sec: f64 },
    /// Status transition (connecting, connected, retrying)
    Status { status: String, message: String },
}

/// Runs the background loop for streaming prices from Polymarket with reconnection support.
pub async fn stream_polymarket_prices_loop<F>(
    token_ids: Vec<String>,
    ws_running: Arc<AtomicBool>,
    on_event: F,
) where
    F: Fn(StreamEvent) + Send + Sync + 'static,
{
    let url = "wss://ws-subscriptions-clob.polymarket.com/ws/market";
    let mut retry_count = 0;
    let max_backoff = 30; // seconds
    let shared_on_event = Arc::new(on_event);

    while ws_running.load(Ordering::SeqCst) {
        if retry_count > 0 {
            let backoff = (2u64.pow(retry_count - 1)).min(max_backoff);
            (shared_on_event)(StreamEvent::Status {
                status: "retrying".to_string(),
                message: format!("Connection lost. Retrying in {}s...", backoff),
            });
            tokio::time::sleep(tokio::time::Duration::from_secs(backoff)).await;
        }

        (shared_on_event)(StreamEvent::Status {
            status: "connecting".to_string(),
            message: "Connecting to Polymarket...".to_string(),
        });

        match connect_async(url).await {
            Ok((ws_stream, _)) => {
                eprintln!("✅ Connected to Polymarket WebSocket");
                (shared_on_event)(StreamEvent::Status {
                    status: "connected".to_string(),
                    message: "Streaming active".to_string(),
                });
                retry_count = 0;

                let (mut write, mut read) = ws_stream.split();
                let (tx, mut rx) = tokio::sync::mpsc::channel::<Message>(32);
                let sender_running = ws_running.clone();

                // Health/Latency state
                let last_pong = Arc::new(tokio::sync::Mutex::new(std::time::Instant::now()));
                let msg_count = Arc::new(std::sync::atomic::AtomicU64::new(0));

                // Sender Task
                tokio::spawn(async move {
                    while sender_running.load(Ordering::SeqCst) {
                        if let Some(msg) = rx.recv().await {
                            if let Err(e) = write.send(msg).await {
                                eprintln!("❌ WS Send Error: {}", e);
                                break;
                            }
                        }
                    }
                });

                // 1. Subscribe
                let subscribe_msg = serde_json::json!({
                    "type": "market",
                    "assets_ids": token_ids,
                });
                let _ = tx
                    .send(Message::Text(subscribe_msg.to_string().into()))
                    .await;

                // 2. Health Monitor Task
                let health_tx = tx.clone();
                let health_running = ws_running.clone();
                let health_pong = last_pong.clone();
                let health_count = msg_count.clone();
                let health_on_event = shared_on_event.clone();

                tokio::spawn(async move {
                    let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(5));
                    while health_running.load(Ordering::SeqCst) {
                        interval.tick().await;

                        // Send PING
                        if (health_tx.send(Message::Ping(Vec::new().into())).await).is_err() {
                            break;
                        }

                        // Calculate and Emit Health
                        let count = health_count.swap(0, Ordering::SeqCst);
                        let latency = health_pong.lock().await.elapsed().as_millis() as u64;

                        (health_on_event)(StreamEvent::Health {
                            latency_ms: if latency > 5000 { 0 } else { latency }, // simplistic ping/pong diff
                            msgs_per_sec: count as f64 / 5.0,
                        });
                    }
                });

                // 3. Listen loop
                let loop_on_event = shared_on_event.clone();
                while ws_running.load(Ordering::SeqCst) {
                    match read.next().await {
                        Some(Ok(msg)) => {
                            match msg {
                                Message::Text(text) => {
                                    msg_count.fetch_add(1, Ordering::SeqCst);
                                    if let Ok(value) =
                                        serde_json::from_str::<serde_json::Value>(&text)
                                    {
                                        let mut handled = false;
                                        if let Some("price_change") =
                                            value.get("event_type").and_then(|v| v.as_str())
                                        {
                                            if let Some(price_changes) = value
                                                .get("price_changes")
                                                .and_then(|v| v.as_array())
                                            {
                                                for change in price_changes {
                                                    let bid =
                                                        parse_price_value(change.get("best_bid"));
                                                    let ask =
                                                        parse_price_value(change.get("best_ask"));
                                                    let price_raw =
                                                        parse_price_value(change.get("price"));
                                                    let price = match (bid, ask) {
                                                        (Some(b), Some(a)) => Some((b + a) / 2.0),
                                                        _ => price_raw,
                                                    };
                                                    if let (Some(p), Some(asset_id)) = (
                                                        price,
                                                        change
                                                            .get("asset_id")
                                                            .and_then(|v| v.as_str()),
                                                    ) {
                                                        (loop_on_event)(StreamEvent::Data {
                                                            asset_id: asset_id.to_string(),
                                                            price: p,
                                                        });
                                                        handled = true;
                                                    }
                                                }
                                            }
                                        }
                                        if !handled && text != "[]" && text != "PONG" {
                                            // Handle other data formats or fallback
                                        }
                                    }
                                }
                                Message::Pong(_) => {
                                    *last_pong.lock().await = std::time::Instant::now();
                                }
                                Message::Close(_) => break,
                                _ => {}
                            }
                        }
                        _ => break, // Connection dropped
                    }
                }

                // If we reach here, connection was lost or closed
                if !ws_running.load(Ordering::SeqCst) {
                    break;
                }
                retry_count += 1;
            }
            Err(e) => {
                eprintln!("❌ Connection error: {}", e);
                retry_count += 1;
            }
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
