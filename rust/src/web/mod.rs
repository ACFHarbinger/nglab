/*!
 * Web scrapers and real-time data streaming.
 *
 * Implements WebSocket clients for live price feeds and HTTP scrapers
 * for historical data ingestion from platforms like Polymarket.
 */

/// Unified exchange interface and adapters.
pub mod exchanges;
/// Polymarket integration and data handling.
pub mod polymarket;
/// Generic web scraping traits and utilities.
pub mod scraper;
/// Real-time data streaming via WebSockets.
pub mod streaming;
