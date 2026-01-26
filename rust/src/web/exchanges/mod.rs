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
pub struct MarketSearchResult {
    pub id: String,
    pub title: String,
    pub description: Option<String>,
    pub outcomes: Vec<String>,
    pub token_ids: Vec<String>,
    pub volume: Option<f64>,
    pub liquidity: Option<f64>,
    pub active: bool,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomerCandle {
    pub time: u64, // Unix timestamp in seconds
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceUpdate {
    pub exchange: String,
    pub symbol: String,
    pub price: f64,
}

#[async_trait]
pub trait Exchange: Send + Sync {
    /// Get the name of the exchange (e.g. "Binance", "Polymarket")
    fn name(&self) -> &str;

    /// Connect/Authenticate
    async fn connect(
        &mut self,
        api_key: Option<String>,
        api_secret: Option<String>,
    ) -> Result<(), Box<dyn Error + Send + Sync>>;

    /// Search for markets
    async fn search_markets(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<MarketSearchResult>, Box<dyn Error + Send + Sync>>;

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

    /// Start streaming prices for specific symbols.
    /// This should run in the background or be an async stream.
    async fn stream_prices(
        &self,
        symbols: Vec<String>,
        on_update: Box<dyn Fn(PriceUpdate) + Send + Sync>,
    ) -> Result<(), Box<dyn Error + Send + Sync>>;
}

pub mod binance;
pub mod manager;
pub mod polymarket_adapter;
