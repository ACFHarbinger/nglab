use super::{
    CustomerCandle, Exchange, MarketData, MarketSearchResult, OrderRequest, OrderResponse,
    PriceUpdate,
};
use async_trait::async_trait;
use reqwest::Client;
use serde::Deserialize;
use serde_json::Value;
use std::collections::HashMap;
use std::error::Error;

pub struct Kraken {
    client: Client,
}

impl Kraken {
    pub fn new() -> Self {
        Self {
            client: Client::new(),
        }
    }
}

// Kraken API Response Wrappers
#[derive(Deserialize, Debug)]
struct KrakenResponse<T> {
    error: Vec<String>,
    result: Option<T>,
}

#[derive(Deserialize, Debug)]
struct KrakenAssetPair {
    altname: String,
    wsname: Option<String>,
    base: String,
    quote: String,
    status: String,
}

#[derive(Deserialize, Debug)]
struct KrakenTickerInfo {
    c: [String; 2], // Close [price, lot_volume]
    b: [String; 3], // Bid [price, whole_lot_volume, lot_volume]
    a: [String; 3], // Ask [price, whole_lot_volume, lot_volume]
    v: [String; 2], // Volume [today, 24h]
}

#[async_trait]
impl Exchange for Kraken {
    fn name(&self) -> &str {
        "Kraken"
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
        let url = "https://api.kraken.com/0/public/AssetPairs";
        let resp: KrakenResponse<HashMap<String, KrakenAssetPair>> =
            self.client.get(url).send().await?.json().await?;

        if let Some(error) = resp.error.first() {
            return Err(format!("Kraken API error: {}", error).into());
        }

        let pairs = resp.result.ok_or("No result from Kraken")?;
        let query_upper = query.to_uppercase();

        let mut results: Vec<MarketSearchResult> = pairs
            .into_iter()
            .filter(|(_k, v)| {
                v.status == "online"
                    && (v.altname.contains(&query_upper)
                        || v.wsname
                            .as_ref()
                            .map_or(false, |w| w.contains(&query_upper)))
            })
            .map(|(k, v)| MarketSearchResult {
                id: k, // Kraken internal name (e.g., XXBTZUSD)
                title: v.wsname.unwrap_or_else(|| v.altname.clone()),
                description: Some(format!("Kraken {}/{}", v.base, v.quote)),
                outcomes: vec![v.base, v.quote],
                token_ids: vec![v.altname],
                volume: None,
                liquidity: None,
                active: true,
            })
            .collect();

        results.truncate(limit);
        Ok(results)
    }

    async fn get_market_data(
        &self,
        symbol: &str,
    ) -> Result<MarketData, Box<dyn Error + Send + Sync>> {
        let url = format!("https://api.kraken.com/0/public/Ticker?pair={}", symbol);
        let resp: KrakenResponse<HashMap<String, KrakenTickerInfo>> =
            self.client.get(&url).send().await?.json().await?;

        if let Some(error) = resp.error.first() {
            return Err(format!("Kraken API error: {}", error).into());
        }

        let tickers = resp.result.ok_or("No result from Kraken")?;
        let ticker = tickers
            .get(symbol)
            .ok_or_else(|| format!("Symbol {} not found", symbol))?;

        let price: f64 = ticker.c[0].parse()?;
        let volume: f64 = ticker.v[1].parse()?;
        let best_bid: f64 = ticker.b[0].parse()?;
        let best_ask: f64 = ticker.a[0].parse()?;

        Ok(MarketData {
            symbol: symbol.to_string(),
            price,
            volume,
            best_bid,
            best_ask,
        })
    }

    async fn place_order(
        &self,
        _order: OrderRequest,
    ) -> Result<OrderResponse, Box<dyn Error + Send + Sync>> {
        Err("Kraken order placement not implemented".into())
    }

    async fn get_historical_candles(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> Result<Vec<CustomerCandle>, Box<dyn Error + Send + Sync>> {
        // Map interval to Kraken minutes
        let kraken_interval = match interval {
            "1m" => 1,
            "5m" => 5,
            "15m" => 15,
            "30m" => 30,
            "1h" => 60,
            "4h" => 240,
            "1d" => 1440,
            "1w" => 10080,
            _ => 60, // Default to 1h
        };

        // Note: Kraken doesn't support 'limit' directly in OHLC, it uses 'since'.
        // We fetch and then truncate.
        let url = format!(
            "https://api.kraken.com/0/public/OHLC?pair={}&interval={}",
            symbol, kraken_interval
        );

        let resp: KrakenResponse<Value> = self.client.get(&url).send().await?.json().await?;

        if let Some(error) = resp.error.first() {
            return Err(format!("Kraken API error: {}", error).into());
        }

        let result_obj = resp.result.ok_or("No result from Kraken")?;
        // Result contains pair name as key (e.g., "XXBTZUSD") and "last" field.
        // We need to find the array.
        let candles_json = result_obj
            .get(symbol)
            .or_else(|| {
                // Sometimes the key might slightly differ or is unknown, try to find the first array
                result_obj
                    .as_object()
                    .and_then(|obj| obj.values().find(|v| v.is_array()))
            })
            .ok_or("Could not find OHLC data in response")?;

        let candles_array = candles_json.as_array().ok_or("OHLC data is not an array")?;

        let mut candles: Vec<CustomerCandle> = Vec::new();

        for item in candles_array.iter().rev().take(limit) {
            if let Some(arr) = item.as_array() {
                if arr.len() >= 6 {
                    let time = arr[0].as_u64().unwrap_or(0);
                    let open: f64 = arr[1].as_str().unwrap_or("0").parse().unwrap_or(0.0);
                    let high: f64 = arr[2].as_str().unwrap_or("0").parse().unwrap_or(0.0);
                    let low: f64 = arr[3].as_str().unwrap_or("0").parse().unwrap_or(0.0);
                    let close: f64 = arr[4].as_str().unwrap_or("0").parse().unwrap_or(0.0);
                    let volume: f64 = arr[6].as_str().unwrap_or("0").parse().unwrap_or(0.0);

                    candles.push(CustomerCandle {
                        time,
                        open,
                        high,
                        low,
                        close,
                        volume,
                    });
                }
            }
        }
        candles.reverse();
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

        let url = "wss://ws.kraken.com";
        let (ws_stream, _) = connect_async(url).await?;
        let (mut write, mut read) = ws_stream.split();

        // Subscribe
        let payload = json!({
            "event": "subscribe",
            "pair": symbols,
            "subscription": {
                "name": "ticker"
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

                // Kraken ticker update is an array: [channelID, {nested data}, channelName, pair]
                if let Some(arr) = v.as_array() {
                    if arr.len() >= 4 {
                        if let (Some(data), Some(pair_val)) = (arr.get(1), arr.last()) {
                            if let (Some(c_arr), Some(pair)) = (data.get("c"), pair_val.as_str()) {
                                if let Some(price_str) = c_arr.get(0).and_then(|v| v.as_str()) {
                                    let price: f64 = price_str.parse().unwrap_or(0.0);
                                    on_update(PriceUpdate {
                                        exchange: "Kraken".to_string(),
                                        symbol: pair.to_string(),
                                        price,
                                    });
                                }
                            }
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
    fn test_kraken_response_deserialization() {
        let json = r#"
        {
            "error": [],
            "result": {
                "XXBTZUSD": {
                    "a": ["96850.00000", "1", "1.000"],
                    "b": ["96849.90000", "5", "5.000"],
                    "c": ["96850.10000", "0.5"],
                    "v": ["1000.0", "5000.0"],
                    "p": ["96000.0", "95500.0"],
                    "t": [100, 500],
                    "l": ["94000.0", "93000.0"],
                    "h": ["97000.0", "98000.0"],
                    "o": "95000.0"
                }
            }
        }
        "#;
        let resp: KrakenResponse<HashMap<String, KrakenTickerInfo>> =
            serde_json::from_str(json).unwrap();
        assert!(resp.error.is_empty());
        let results = resp.result.as_ref().unwrap();
        let info = results.get("XXBTZUSD").unwrap();
        assert_eq!(info.c[0], "96850.10000");
    }
}
