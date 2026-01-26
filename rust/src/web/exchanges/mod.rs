use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::error::Error;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketData {
    pub symbol: String,
    pub price: f64,
    pub volume: f64,
    pub best_bid: f64,
    pub best_ask: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderRequest {
    pub symbol: String,
    pub side: String, // "buy" or "sell"
    pub quantity: f64,
    pub price: Option<f64>, // None for market orders
    pub order_type: String, // "limit", "market", etc.
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderResponse {
    pub order_id: String,
    pub status: String,
}

#[async_trait]
pub trait Exchange: Send + Sync {
    /// Get the name of the exchange (e.g. "Binance", "Kraken")
    fn name(&self) -> &str;

    /// Connect/Authenticate
    async fn connect(
        &mut self,
        api_key: Option<String>,
        api_secret: Option<String>,
    ) -> Result<(), Box<dyn Error + Send + Sync>>;

    /// Fetch current market data for a symbol
    async fn get_market_data(
        &self,
        symbol: &str,
    ) -> Result<MarketData, Box<dyn Error + Send + Sync>>;

    /// Place an order
    async fn place_order(
        &self,
        order: OrderRequest,
    ) -> Result<OrderResponse, Box<dyn Error + Send + Sync>>;

    /// Get historical candles (OHLCV)
    async fn get_historical_candles(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> Result<Vec<CustomerCandle>, Box<dyn Error + Send + Sync>>;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomerCandle {
    pub time: u64, // Unix timestamp in seconds
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

pub mod binance;
