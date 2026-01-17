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
pub fn resolve_polymarket_token_ids(market_source: &str) -> Result<Vec<String>, String> {
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
        .map_err(|e| format!("Failed to resolve market: {}", e))?;

    let metadata = scraper.get_metadata();
    Ok(metadata.outcomes.into_iter().map(|o| o.id).collect())
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
            let (mut write, mut read) = ws_stream.split();

            // Subscribe (Polymarket WebSocket format)
            let subscribe_msg = serde_json::json!({
                "type": "market",
                "assets_ids": token_ids,
            });

            if let Err(e) = write
                .send(Message::Text(subscribe_msg.to_string().into()))
                .await
            {
                eprintln!("❌ Failed to send Polymarket subscription: {}", e);
                ws_running.store(false, Ordering::SeqCst);
                return;
            }

            // Listen loop
            while ws_running.load(Ordering::SeqCst) {
                if let Some(msg) = read.next().await {
                    match msg {
                        Ok(Message::Text(text)) => {
                            if let Ok(value) = serde_json::from_str::<serde_json::Value>(&text) {
                                let mut handled = false;

                                // Format 0: Official Polymarket price_change event
                                if let Some(event_type) =
                                    value.get("event_type").and_then(|s| s.as_str())
                                {
                                    if event_type == "price_change" {
                                        if let Some(price_changes) =
                                            value.get("price_changes").and_then(|v| v.as_array())
                                        {
                                            for change in price_changes {
                                                if let Some(asset_id) =
                                                    change.get("asset_id").and_then(|s| s.as_str())
                                                {
                                                    let bid_opt =
                                                        parse_price_value(change.get("best_bid"));
                                                    let ask_opt =
                                                        parse_price_value(change.get("best_ask"));

                                                    if let (Some(bid), Some(ask)) =
                                                        (bid_opt, ask_opt)
                                                    {
                                                        let price = (bid + ask) / 2.0;
                                                        on_update(PolymarketPriceUpdate {
                                                            asset_id: asset_id.to_string(),
                                                            price,
                                                        });
                                                        handled = true;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }

                                // Format 1: Direct array of updates
                                if !handled {
                                    if let Some(arr) = value.as_array() {
                                        for item in arr {
                                            if let Some(asset_id) =
                                                item.get("asset_id").and_then(|s| s.as_str())
                                            {
                                                if let Some(price) =
                                                    parse_price_value(item.get("price"))
                                                {
                                                    on_update(PolymarketPriceUpdate {
                                                        asset_id: asset_id.to_string(),
                                                        price,
                                                    });
                                                    handled = true;
                                                }
                                            }
                                        }
                                    }
                                }

                                // Format 2: Single object update
                                if !handled {
                                    if let Some(asset_id) =
                                        value.get("asset_id").and_then(|s| s.as_str())
                                    {
                                        if let Some(price) = parse_price_value(value.get("price")) {
                                            on_update(PolymarketPriceUpdate {
                                                asset_id: asset_id.to_string(),
                                                price,
                                            });
                                            handled = true;
                                        }
                                    }
                                }

                                // Format 3: Nested data structure
                                if !handled {
                                    if let Some(data) = value.get("data") {
                                        if let Some(arr) = data.as_array() {
                                            for item in arr {
                                                if let Some(asset_id) =
                                                    item.get("asset_id").and_then(|s| s.as_str())
                                                {
                                                    if let Some(price) =
                                                        parse_price_value(item.get("price"))
                                                    {
                                                        on_update(PolymarketPriceUpdate {
                                                            asset_id: asset_id.to_string(),
                                                            price,
                                                        });
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        Ok(Message::Ping(data)) => {
                            let _ = write.send(Message::Pong(data)).await;
                        }
                        Ok(Message::Close(_)) => break,
                        Err(_) => break,
                        _ => {}
                    }
                } else {
                    break;
                }
            }
            ws_running.store(false, Ordering::SeqCst);
        }
        Err(e) => {
            eprintln!("Failed to connect to Polymarket WS: {}", e);
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
