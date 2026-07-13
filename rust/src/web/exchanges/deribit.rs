use super::{
    CustomerCandle, Exchange, MarketData, MarketSearchResult, OrderRequest, OrderResponse,
    PriceUpdate,
};
use async_trait::async_trait;
use reqwest::Client;
use serde::Deserialize;
use serde_json::Value; // For generic JSON handling
use std::error::Error;

pub struct Deribit {
    client: Client,
}

impl Deribit {
    pub fn new() -> Self {
        Self {
            client: Client::new(),
        }
    }
}

// Deribit API Response Wrappers
#[derive(Deserialize, Debug)]
struct DeribitResponse<T> {
    result: T,
}

#[derive(Deserialize, Debug)]
struct DeribitInstrument {
    instrument_name: String,
    base_currency: String,
    quote_currency: String,
    is_active: bool,
    kind: String, // "future", "option", "spot"
}

#[derive(Deserialize, Debug)]
struct DeribitTicker {
    instrument_name: String,
    last_price: f64,
    best_bid_price: f64,
    best_ask_price: f64,
    stats: DeribitStats,
}

#[derive(Deserialize, Debug)]
struct DeribitStats {
    volume: f64, // 24h volume
}

#[derive(Deserialize, Debug)]
struct DeribitOHLC {
    ticks: Vec<u64>, // Time
    open: Vec<f64>,
    high: Vec<f64>,
    low: Vec<f64>,
    close: Vec<f64>,
    volume: Vec<f64>,
}

#[async_trait]
impl Exchange for Deribit {
    fn name(&self) -> &str {
        "Deribit"
    }

    async fn connect(
        &mut self,
        _api_key: Option<String>,
        _api_secret: Option<String>,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        Ok(())
    }

    async fn search_markets(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<MarketSearchResult>, Box<dyn Error + Send + Sync>> {
        // Deribit requires fetching by currency for instruments (BTC, ETH, USDC, etc.)
        // For broad search, we might fallback to BTC and ETH default, or try to be smart.
        // Let's assume we search BTC instruments by default or infer from query.
        let currency = if query.to_uppercase().contains("ETH") {
            "ETH"
        } else if query.to_uppercase().contains("SOL") {
            "SOL"
        } else {
            "BTC"
        };

        let url = format!(
            "https://www.deribit.com/api/v2/public/get_instruments?currency={}",
            currency
        );

        let resp: DeribitResponse<Vec<DeribitInstrument>> =
            self.client.get(&url).send().await?.json().await?;

        let query_upper = query.to_uppercase();
        let results: Vec<MarketSearchResult> = resp
            .result
            .into_iter()
            .filter(|i| i.is_active && i.instrument_name.contains(&query_upper))
            .take(limit)
            .map(|i| MarketSearchResult {
                id: i.instrument_name.clone(),
                title: i.instrument_name.clone(),
                description: Some(format!("Deribit {} {}", i.kind, i.instrument_name)),
                outcomes: vec![i.base_currency, i.quote_currency],
                token_ids: vec![i.instrument_name],
                volume: None,
                liquidity: None,
                active: true,
            })
            .collect();

        Ok(results)
    }

    async fn get_market_data(
        &self,
        symbol: &str,
    ) -> Result<MarketData, Box<dyn Error + Send + Sync>> {
        let url = format!(
            "https://www.deribit.com/api/v2/public/ticker?instrument_name={}",
            symbol
        );
        let resp: DeribitResponse<DeribitTicker> =
            self.client.get(&url).send().await?.json().await?;

        let t = resp.result;

        Ok(MarketData {
            symbol: symbol.to_string(),
            price: t.last_price,
            volume: t.stats.volume,
            best_bid: t.best_bid_price,
            best_ask: t.best_ask_price,
        })
    }

    async fn place_order(
        &self,
        _order: OrderRequest,
    ) -> Result<OrderResponse, Box<dyn Error + Send + Sync>> {
        Err("Deribit order placement not implemented".into())
    }

    async fn get_historical_candles(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> Result<Vec<CustomerCandle>, Box<dyn Error + Send + Sync>> {
        let resolution = match interval {
            "1m" => "1",
            "5m" => "5",
            "15m" => "15",
            "30m" => "30",
            "1h" => "60",
            "4h" => "240",
            "1d" => "1D",
            _ => "60",
        };

        // Deribit needs start/end. Let's fetch last N periods.
        // Approx time calculation
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        let interval_ms: u64 = match interval {
            "1m" => 60 * 1000,
            "1h" => 60 * 60 * 1000,
            "1d" => 24 * 60 * 60 * 1000,
            _ => 60 * 60 * 1000,
        };

        let start = now - (limit as u64 * interval_ms);

        let url = format!(
            "https://www.deribit.com/api/v2/public/get_tradingview_chart_data?instrument_name={}&start_timestamp={}&end_timestamp={}&resolution={}",
            symbol, start, now, resolution
        );

        let resp: DeribitResponse<DeribitOHLC> = self.client.get(&url).send().await?.json().await?;

        let d = resp.result;
        let mut candles = Vec::new();

        // Deribit returns parallel arrays
        for i in 0..d.ticks.len() {
            candles.push(CustomerCandle {
                time: d.ticks[i] / 1000, // Ms to seconds
                open: d.open[i],
                high: d.high[i],
                low: d.low[i],
                close: d.close[i],
                volume: d.volume[i],
            });
        }
        // Usually delivered in time ASC order.
        Ok(candles)
    }

    async fn stream_prices(
        &self,
        symbols: Vec<String>,
        on_update: Box<dyn Fn(PriceUpdate) + Send + Sync>,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        use futures_util::{SinkExt, StreamExt};
        use serde_json::json;
        use tokio_tungstenite::connect_async;

        if symbols.is_empty() {
            return Ok(());
        }

        let url = "wss://www.deribit.com/ws/api/v2";
        let (ws_stream, _) = connect_async(url).await?;
        let (mut write, mut read) = ws_stream.split();

        // Channels: ticker.instrument_name.100ms
        let channels: Vec<String> = symbols
            .iter()
            .map(|s| format!("ticker.{}.100ms", s))
            .collect();

        let payload = json!({
            "jsonrpc": "2.0",
            "method": "public/subscribe",
            "params": {
                "channels": channels
            }
        });

        write
            .send(tokio_tungstenite::tungstenite::Message::Text(
                payload.to_string().into(),
            ))
            .await?;

        while let Some(Ok(msg)) = read.next().await {
            if let tokio_tungstenite::tungstenite::Message::Text(text) = msg {
                let v: Value = serde_json::from_str(&text)?;

                // Notification structure
                if let Some(params) = v.get("params") {
                    if let (Some(data), Some(channel)) = (
                        params.get("data"),
                        params.get("channel").and_then(|v| v.as_str()),
                    ) {
                        if let Some(price) = data.get("last_price").and_then(|v| v.as_f64()) {
                            // Channel "ticker.BTC-PERPETUAL.100ms"
                            // Extract symbol from channel or data
                            let symbol = data
                                .get("instrument_name")
                                .and_then(|v| v.as_str())
                                .unwrap_or(channel);

                            on_update(PriceUpdate {
                                exchange: "Deribit".to_string(),
                                symbol: symbol.to_string(),
                                price,
                            });
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deribit_instrument_deserialization() {
        let json = r#"
        {
            "result": [
                {
                    "instrument_name": "BTC-PERPETUAL",
                    "base_currency": "BTC",
                    "quote_currency": "USD",
                    "is_active": true,
                    "kind": "future"
                }
            ]
        }
        "#;
        let resp: DeribitResponse<Vec<DeribitInstrument>> = serde_json::from_str(json).unwrap();
        assert_eq!(resp.result[0].instrument_name, "BTC-PERPETUAL");
        assert!(resp.result[0].is_active);
    }
}
