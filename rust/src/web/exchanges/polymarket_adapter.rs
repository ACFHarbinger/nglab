use super::{
    CustomerCandle, Exchange, MarketData, MarketSearchResult, OrderRequest, OrderResponse,
};
use crate::web::polymarket::PolymarketScraper;
use async_trait::async_trait;
use std::error::Error;

pub struct PolymarketAdapter {
    scraper: PolymarketScraper,
}

impl PolymarketAdapter {
    pub fn new() -> Self {
        Self {
            scraper: PolymarketScraper::new(),
        }
    }
}

#[async_trait]
impl Exchange for PolymarketAdapter {
    fn name(&self) -> &str {
        "Polymarket"
    }

    async fn connect(
        &mut self,
        _api_key: Option<String>,
        _api_secret: Option<String>,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        // Polymarket uses public API for most things, or clob-client for trading.
        // For now, we assume public access through the scraper.
        Ok(())
    }

    async fn search_markets(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<MarketSearchResult>, Box<dyn Error + Send + Sync>> {
        let results = self.scraper.search_markets(query, limit).await?;

        Ok(results
            .into_iter()
            .map(|r| MarketSearchResult {
                id: r.id,
                title: r.title,
                description: Some(r.question),
                outcomes: r.outcomes,
                token_ids: r.clob_token_ids,
                volume: r.volume,
                liquidity: r.liquidity,
                active: r.active,
            })
            .collect())
    }

    async fn get_market_data(
        &self,
        symbol: &str,
    ) -> Result<MarketData, Box<dyn Error + Send + Sync>> {
        // Usually Polymarket market data is fetched via resolve_market or streaming.
        // This is a simplified fetch for a single market question/id.
        let mut scraper = PolymarketScraper::new();
        scraper.resolve_market(symbol).await?;
        let meta = scraper.get_metadata();

        // This doesn't actually give price.
        // Real price comes from streaming or clob-api.
        // For now returning mock/placeholder if we don't have a direct "get_price" method in scraper.
        Ok(MarketData {
            symbol: symbol.to_string(),
            price: 0.5, // Placeholder
            volume: 0.0,
            best_bid: 0.49,
            best_ask: 0.51,
        })
    }

    async fn place_order(
        &self,
        _order: OrderRequest,
    ) -> Result<OrderResponse, Box<dyn Error + Send + Sync>> {
        Err("Order placement not implemented for Polymarket adapter yet (requires Poly wallet integration)".into())
    }

    async fn get_historical_candles(
        &self,
        _symbol: &str,
        _interval: &str,
        _limit: usize,
    ) -> Result<Vec<CustomerCandle>, Box<dyn Error + Send + Sync>> {
        // PolymarketScraper handles history but returns raw items.
        // We'd need to convert them to CustomerCandle.
        Ok(Vec::new())
    }

    async fn stream_prices(
        &self,
        symbols: Vec<String>,
        on_update: Box<dyn Fn(super::PriceUpdate) + Send + Sync>,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        use crate::web::streaming::stream_polymarket_prices_loop;
        use std::sync::atomic::AtomicBool;
        use std::sync::Arc;

        let running = Arc::new(AtomicBool::new(true));

        // This will block/run until stopped.
        // In a real implementation we might want to return the handle.
        // But for the trait we'll just implement the loop.

        stream_polymarket_prices_loop(symbols, running, move |update| {
            on_update(super::PriceUpdate {
                exchange: "Polymarket".to_string(),
                symbol: update.asset_id,
                price: update.price,
            });
        })
        .await;

        Ok(())
    }
}
