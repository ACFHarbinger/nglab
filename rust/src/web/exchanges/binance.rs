use super::{
    CustomerCandle, Exchange, MarketData, MarketSearchResult, OrderRequest, OrderResponse,
};
use async_trait::async_trait;
use reqwest::Client;
use serde::Deserialize;
use std::error::Error;

pub struct Binance {
    client: Client,
}

impl Binance {
    pub fn new() -> Self {
        Self {
            client: Client::new(),
        }
    }
}

#[derive(Deserialize)]
struct BinanceTicker {
    symbol: String,
    price: String,
}

#[derive(Deserialize)]
struct BinanceExchangeInfo {
    symbols: Vec<BinanceSymbol>,
}

#[derive(Deserialize)]
struct BinanceSymbol {
    symbol: String,
    status: String,
    #[serde(rename = "baseAsset")]
    base_asset: String,
    #[serde(rename = "quoteAsset")]
    quote_asset: String,
}

#[async_trait]
impl Exchange for Binance {
    fn name(&self) -> &str {
        "Binance"
    }

    async fn connect(
        &mut self,
        _api_key: Option<String>,
        _api_secret: Option<String>,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        // Public API doesn't strictly need API keys for market data.
        // For private endpoints, we'd store them here.
        Ok(())
    }

    async fn search_markets(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<MarketSearchResult>, Box<dyn Error + Send + Sync>> {
        let url = "https://api.binance.com/api/v3/exchangeInfo";
        let info: BinanceExchangeInfo = self.client.get(url).send().await?.json().await?;

        let query_upper = query.to_uppercase();
        let results = info
            .symbols
            .into_iter()
            .filter(|s| {
                s.status == "TRADING"
                    && (s.symbol.contains(&query_upper) || s.base_asset.contains(&query_upper))
            })
            .take(limit)
            .map(|s| MarketSearchResult {
                id: s.symbol.clone(),
                title: format!("{}/{}", s.base_asset, s.quote_asset),
                description: Some(format!("Binance {}/{}", s.base_asset, s.quote_asset)),
                outcomes: vec![s.base_asset, s.quote_asset],
                token_ids: vec![s.symbol],
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
            "https://api.binance.com/api/v3/ticker/price?symbol={}",
            symbol
        );
        let ticker: BinanceTicker = self.client.get(&url).send().await?.json().await?;
        let price: f64 = ticker.price.parse()?;

        Ok(MarketData {
            symbol: symbol.to_string(),
            price,
            volume: 0.0,
            best_bid: price - 0.01, // Ticker/price doesn't give spreads. Would need /ticker/bookTicker
            best_ask: price + 0.01,
        })
    }

    async fn place_order(
        &self,
        order: OrderRequest,
    ) -> Result<OrderResponse, Box<dyn Error + Send + Sync>> {
        // Institutional-grade simulation: In a real environment, we'd sign the request
        // and hit /api/v3/order. For now, we simulate success for "paper trading".

        tracing::info!(
            "Binance Paper Order: {} {} units of {} @ {:?}",
            order.side,
            order.quantity,
            order.symbol,
            order.price
        );

        // Generate a pseudo-random order ID
        let order_id = format!(
            "bin-sim-{}",
            uuid::Uuid::new_v4()
                .to_string()
                .get(..8)
                .unwrap_or("unknown")
        );

        Ok(OrderResponse {
            order_id,
            status: "FILLED".to_string(), // In paper mode, we assume immediate fill for simplicity
        })
    }

    async fn get_historical_candles(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> Result<Vec<CustomerCandle>, Box<dyn Error + Send + Sync>> {
        let url = format!(
            "https://api.binance.com/api/v3/klines?symbol={}&interval={}&limit={}",
            symbol, interval, limit
        );

        let response: Vec<Vec<serde_json::Value>> =
            self.client.get(&url).send().await?.json().await?;

        let candles = response
            .into_iter()
            .map(|item| {
                CustomerCandle {
                    time: item[0].as_u64().unwrap_or(0) / 1000, // Ms to seconds
                    open: item[1].as_str().unwrap_or("0").parse().unwrap_or(0.0),
                    high: item[2].as_str().unwrap_or("0").parse().unwrap_or(0.0),
                    low: item[3].as_str().unwrap_or("0").parse().unwrap_or(0.0),
                    close: item[4].as_str().unwrap_or("0").parse().unwrap_or(0.0),
                    volume: item[5].as_str().unwrap_or("0").parse().unwrap_or(0.0),
                }
            })
            .collect();

        Ok(candles)
    }

    async fn stream_prices(
        &self,
        symbols: Vec<String>,
        on_update: Box<dyn Fn(super::PriceUpdate) + Send + Sync>,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        use futures_util::StreamExt;
        use serde_json::Value;
        use tokio_tungstenite::connect_async;

        if symbols.is_empty() {
            return Ok(());
        }

        // Multiple symbols: /stream?streams=btcusdt@ticker/ethusdt@ticker
        let streams = symbols
            .iter()
            .map(|s| format!("{}@ticker", s.to_lowercase()))
            .collect::<Vec<_>>()
            .join("/");
        let url = format!("wss://stream.binance.com:9443/stream?streams={}", streams);

        let (ws_stream, _) = connect_async(url).await?;
        let (_, mut read) = ws_stream.split();

        while let Some(Ok(msg)) = read.next().await {
            if let tokio_tungstenite::tungstenite::Message::Text(text) = msg {
                let v: Value = serde_json::from_str(&text)?;

                if let Some(data) = v.get("data") {
                    if let (Some(s), Some(c)) = (
                        data.get("s").and_then(|v| v.as_str()),
                        data.get("c").and_then(|v| v.as_str()),
                    ) {
                        let price: f64 = c.parse().unwrap_or(0.0);
                        on_update(super::PriceUpdate {
                            exchange: "Binance".to_string(),
                            symbol: s.to_string(),
                            price,
                        });
                    }
                }
            }
        }

        Ok(())
    }
}
