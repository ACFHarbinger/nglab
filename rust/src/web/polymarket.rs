/*!
 * Polymarket data scraper.
 *
 * Fetches market metadata and historical price data
 * from the Polymarket API.
 */

use crate::errors::{ArenaError, ArenaResult};
use crate::web::scraper::WebScraper;
use chrono::{DateTime, Utc};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/**
 * Data frequency for historical price fetching.
 */
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Frequency {
    Minutely,
    Hourly,
    Daily,
    Weekly,
    Monthly,
}

impl Frequency {
    fn to_interval(&self) -> (&'static str, i64) {
        match self {
            Frequency::Minutely => ("1m", 1),
            Frequency::Hourly => ("1h", 60),
            Frequency::Daily => ("1d", 1440),
            Frequency::Weekly => ("1w", 10080),
            Frequency::Monthly => ("30d", 43200), // Approx
        }
    }
}

/**
 * Information about a specific market outcome (e.g., "Yes", "No").
 */
#[derive(Serialize, Clone, Debug)]
pub struct OutcomeInfo {
    pub id: String,
    pub name: String,
}

/**
 * Metadata for a Polymarket prediction market.
 */
#[derive(Serialize, Clone, Debug)]
pub struct MarketMetadata {
    pub title: String,
    pub outcomes: Vec<OutcomeInfo>,
}

/**
 * Scraper for fetching and processing Polymarket data.
 */
pub struct PolymarketScraper {
    client: Client,
    token_ids: Vec<String>,
    token_names: HashMap<String, String>,
    market_title: Option<String>,
    start_date: Option<DateTime<Utc>>,
    end_date: Option<DateTime<Utc>>,
    frequency: Frequency,
}

#[derive(Deserialize, Debug)]
struct HistoryResponse {
    history: Vec<HistoryItem>,
}

#[derive(Deserialize, Debug)]
struct HistoryItem {
    t: u64,
    p: f64,
}

impl PolymarketScraper {
    pub fn new() -> Self {
        PolymarketScraper {
            client: Client::new(),
            token_ids: Vec::new(),
            token_names: HashMap::new(),
            market_title: None,
            start_date: None,
            end_date: None,
            frequency: Frequency::Daily,
        }
    }

    pub fn with_frequency(mut self, frequency: Frequency) -> Self {
        self.frequency = frequency;
        self
    }

    pub fn with_date_range(mut self, start: DateTime<Utc>, end: DateTime<Utc>) -> Self {
        self.start_date = Some(start);
        self.end_date = Some(end);
        self
    }

    pub fn with_option(mut self, token_id: &str) -> Self {
        self.token_ids.push(token_id.to_string());
        self
    }

    pub fn with_options(mut self, token_ids: Vec<String>) -> Self {
        self.token_ids.extend(token_ids);
        self
    }

    pub fn filter_options(mut self, selected_ids: Vec<String>) -> Self {
        self.token_ids = selected_ids;
        self
    }

    pub fn get_metadata(&self) -> MarketMetadata {
        let mut outcomes = Vec::new();
        // Maintain order from token_ids
        for id in &self.token_ids {
            let name = self
                .token_names
                .get(id)
                .cloned()
                .unwrap_or_else(|| format!("Price_{}", id));
            outcomes.push(OutcomeInfo {
                id: id.clone(),
                name,
            });
        }

        MarketMetadata {
            title: self
                .market_title
                .clone()
                .unwrap_or_else(|| "Unknown Market".to_string()),
            outcomes,
        }
    }

    /**
     * Resolve a market or event by ID or slug and populate metadata.
     *
     * @param market_id The Polymarket ID or human-readable slug.
     */
    pub async fn resolve_market(&mut self, market_id: &str) -> ArenaResult<()> {
        let url = format!("https://gamma-api.polymarket.com/markets/{}", market_id);

        // Try to fetch as a single market first
        let response = self.client.get(&url).send().await?.error_for_status();

        match response {
            Ok(resp) => {
                let market: MarketResponse = resp.json().await?;
                self.process_market(market);
                Ok(())
            }
            Err(_) => {
                // If failed, maybe it's an Event ID? Try events endpoint
                let url = format!("https://gamma-api.polymarket.com/events?id={}", market_id);
                if let Ok(response) = self.client.get(&url).send().await?.error_for_status() {
                    let events: Vec<EventResponse> = response.json().await?;
                    self.process_events(events)?;
                    Ok(())
                } else {
                    // Finally, try as a SLUG (Event first, then Market)
                    let url = format!("https://gamma-api.polymarket.com/events?slug={}", market_id);
                    if let Ok(response) = self.client.get(&url).send().await?.error_for_status() {
                        let events: Vec<EventResponse> = response.json().await?;
                        if !events.is_empty() {
                            self.process_events(events)?;
                            return Ok(());
                        }
                    }

                    // If not event slug, maybe Market Slug?
                    let url = format!(
                        "https://gamma-api.polymarket.com/markets?slug={}",
                        market_id
                    );
                    if let Ok(response) = self.client.get(&url).send().await?.error_for_status() {
                        let markets: Vec<MarketResponse> = response.json().await?;
                        if !markets.is_empty() {
                            for market in markets {
                                self.process_market(market);
                            }
                            return Ok(());
                        }
                    }

                    Err(ArenaError::InvalidOrder(format!(
                        "Market/Event ID or Slug '{}' not found",
                        market_id
                    )))
                }
            }
        }
    }

    fn process_events(&mut self, events: Vec<EventResponse>) -> ArenaResult<()> {
        if let Some(event) = events.first() {
            // Use Event Title
            if self.market_title.is_none() {
                self.market_title = Some(event.title.clone());
            }

            for market in &event.markets {
                self.process_market(market.clone());
            }
            Ok(())
        } else {
            Err(ArenaError::InvalidOrder(
                "Event found but no markets".to_string(),
            ))
        }
    }

    fn process_market(&mut self, market: MarketResponse) {
        // Use Market Question if we don't have a title yet
        if self.market_title.is_none() {
            self.market_title = Some(market.question.clone());
        }

        // clobTokenIds is often a JSON string stringified, e.g. "[\"0x...\",\"0x...\"]"
        // We need to parse it safely.
        let ids: Vec<String> = serde_json::from_str(&market.clob_token_ids).unwrap_or_default();
        let outcomes: Vec<String> = serde_json::from_str(&market.outcomes).unwrap_or_default();

        for (i, token_id) in ids.iter().enumerate() {
            self.token_ids.push(token_id.clone());
            if let Some(name) = outcomes.get(i) {
                // Heuristic: If outcomes are "Yes"/"No", use the Market Question to disambiguate.
                // E.g. "Will Trump win?" -> Yes (Name directly) is bad.
                // Better: "Will Trump win? (Yes)"
                let final_name = if outcomes.len() == 2
                    && outcomes.contains(&"Yes".to_string())
                    && outcomes.contains(&"No".to_string())
                {
                    if name == "Yes" {
                        market.question.clone()
                    } else {
                        format!("No - {}", market.question)
                    }
                } else {
                    name.clone()
                };

                self.token_names.insert(token_id.clone(), final_name);
            }
        }
    }

    async fn fetch_history(&self, token_id: &str) -> ArenaResult<Vec<HistoryItem>> {
        let (_, fidelity) = self.frequency.to_interval();

        let fidelity_str = fidelity.to_string();
        let mut params = vec![("market", token_id), ("fidelity", &fidelity_str)];

        let start_ts_str;
        let end_ts_str;
        let has_dates = self.start_date.is_some() || self.end_date.is_some();

        if has_dates {
            if let Some(start) = self.start_date {
                start_ts_str = start.timestamp().to_string();
                params.push(("startTs", &start_ts_str));
            }
            if let Some(end) = self.end_date {
                end_ts_str = end.timestamp().to_string();
                params.push(("endTs", &end_ts_str));
            }
        } else {
            // If no dates provided, fetch MAX history
            params.push(("interval", "max"));
        }

        let url = "https://clob.polymarket.com/prices-history";

        // Use ? to propagate reqwest::Error converted to ArenaError::Network
        let response = self
            .client
            .get(url)
            .query(&params)
            .send()
            .await?
            .error_for_status()?;

        let data: HistoryResponse = response.json().await?;

        Ok(data.history)
    }

    pub async fn get_trending_events(&self, limit: usize) -> ArenaResult<Vec<EventResponse>> {
        let url = "https://gamma-api.polymarket.com/events";
        let params = [
            ("limit", limit.to_string()),
            ("sort", "volume".to_string()),
            ("order", "desc".to_string()),
        ];

        let response = self
            .client
            .get(url)
            .query(&params)
            .send()
            .await?
            .error_for_status()?;

        let events: Vec<EventResponse> = response.json().await?;
        Ok(events)
    }

    /// Search for open/active markets by query string.
    ///
    /// Searches the Polymarket gamma API for events matching the query,
    /// filters to only active/open markets, and returns structured results.
    pub async fn search_markets(
        &self,
        query: &str,
        limit: usize,
    ) -> ArenaResult<Vec<MarketSearchResult>> {
        let url = "https://gamma-api.polymarket.com/events";

        // Build query params - search by title_contains and filter to active
        let params = [
            ("limit", limit.to_string()),
            ("active", "true".to_string()),
            ("closed", "false".to_string()),
            ("title_contains", query.to_string()),
            ("order", "volume".to_string()),
            ("ascending", "false".to_string()),
        ];

        let response = self
            .client
            .get(url)
            .query(&params)
            .send()
            .await?
            .error_for_status()?;

        let events: Vec<SearchEventResponse> = response.json().await?;

        // Transform events into search results
        let mut results = Vec::new();
        for event in events {
            for market in event.markets {
                // Parse outcomes and token IDs from stringified JSON
                let outcomes: Vec<String> =
                    serde_json::from_str(&market.outcomes).unwrap_or_default();
                let clob_token_ids: Vec<String> =
                    serde_json::from_str(&market.clob_token_ids).unwrap_or_default();

                // Only include markets with valid token IDs (tradeable)
                if clob_token_ids.is_empty() {
                    continue;
                }

                results.push(MarketSearchResult {
                    id: market.id,
                    event_id: event.id.clone(),
                    title: event.title.clone(),
                    question: market.question,
                    outcomes,
                    clob_token_ids,
                    volume: market.volume_num,
                    liquidity: market.liquidity_num,
                    end_date: market.end_date_iso,
                    active: market.active.unwrap_or(true),
                });
            }
        }

        Ok(results)
    }

    /// Get open/active markets without search query (for browsing).
    pub async fn get_open_markets(&self, limit: usize) -> ArenaResult<Vec<MarketSearchResult>> {
        let url = "https://gamma-api.polymarket.com/events";

        let params = [
            ("limit", limit.to_string()),
            ("active", "true".to_string()),
            ("closed", "false".to_string()),
            ("order", "volume".to_string()),
            ("ascending", "false".to_string()),
        ];

        let response = self
            .client
            .get(url)
            .query(&params)
            .send()
            .await?
            .error_for_status()?;

        let events: Vec<SearchEventResponse> = response.json().await?;

        let mut results = Vec::new();
        for event in events {
            for market in event.markets {
                let outcomes: Vec<String> =
                    serde_json::from_str(&market.outcomes).unwrap_or_default();
                let clob_token_ids: Vec<String> =
                    serde_json::from_str(&market.clob_token_ids).unwrap_or_default();

                if clob_token_ids.is_empty() {
                    continue;
                }

                results.push(MarketSearchResult {
                    id: market.id,
                    event_id: event.id.clone(),
                    title: event.title.clone(),
                    question: market.question,
                    outcomes,
                    clob_token_ids,
                    volume: market.volume_num,
                    liquidity: market.liquidity_num,
                    end_date: market.end_date_iso,
                    active: market.active.unwrap_or(true),
                });
            }
        }

        Ok(results)
    }
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct MarketResponse {
    pub id: String,
    pub question: String,
    pub outcomes: String, // Stringified JSON array: "[\"Yes\", \"No\"]"
    #[serde(rename = "clobTokenIds")]
    pub clob_token_ids: String, // Stringified JSON array
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct EventResponse {
    pub id: String,
    pub title: String,
    pub markets: Vec<MarketResponse>,
}

/// Search result for a market, including additional display info.
#[derive(Serialize, Debug, Clone)]
pub struct MarketSearchResult {
    pub id: String,
    pub event_id: String,
    pub title: String,
    pub question: String,
    pub outcomes: Vec<String>,
    pub clob_token_ids: Vec<String>,
    pub volume: Option<f64>,
    pub liquidity: Option<f64>,
    pub end_date: Option<String>,
    pub active: bool,
}

/// Extended market response for search results with optional fields.
#[derive(Deserialize, Debug, Clone, Default)]
#[serde(default)]
struct SearchMarketResponse {
    id: String,
    question: String,
    #[serde(default)]
    outcomes: String,
    #[serde(rename = "clobTokenIds", default)]
    clob_token_ids: String,
    #[serde(rename = "volumeNum")]
    volume_num: Option<f64>,
    #[serde(rename = "liquidityNum")]
    liquidity_num: Option<f64>,
    #[serde(rename = "endDateIso")]
    end_date_iso: Option<String>,
    #[serde(default)]
    active: Option<bool>,
}

/// Extended event response for search results.
#[derive(Deserialize, Debug)]
struct SearchEventResponse {
    id: String,
    title: String,
    markets: Vec<SearchMarketResponse>,
}

impl WebScraper for PolymarketScraper {
    async fn download_csv(&self, output_path: &str) -> ArenaResult<()> {
        if self.token_ids.is_empty() {
            return Err(ArenaError::InvalidOrder(
                "No options (token IDs) selected".to_string(),
            ));
        }

        // Fetch data for all options
        let mut all_histories = HashMap::new();
        let mut all_timestamps = Vec::new();

        for token_id in &self.token_ids {
            let history = self.fetch_history(token_id).await?;
            for item in &history {
                all_timestamps.push(item.t);
            }
            all_histories.insert(token_id.clone(), history);
        }

        all_timestamps.sort_unstable();
        all_timestamps.dedup();

        // Write CSV
        let mut wtr = csv::Writer::from_path(output_path)?;

        let mut headers = vec!["Date (UTC)".to_string(), "Timestamp (UTC)".to_string()];
        for token_id in &self.token_ids {
            // Use name if available, else "Price_{ID}"
            let header = if let Some(name) = self.token_names.get(token_id) {
                name.clone()
            } else {
                format!("Price_{}", token_id)
            };
            headers.push(header);
        }
        wtr.write_record(&headers)?;

        for ts in all_timestamps {
            let mut row = Vec::new();

            // Date string
            if let Some(datetime) = DateTime::from_timestamp(ts as i64, 0) {
                row.push(datetime.format("%d-%m-%Y %H:%M").to_string());
            } else {
                row.push("Invalid Timestamp".to_string());
            }

            // Timestamp
            row.push(ts.to_string());

            // Prices
            for token_id in &self.token_ids {
                let history = all_histories.get(token_id).unwrap();
                // Find price at this timestamp
                if let Some(item) = history.iter().find(|h| h.t == ts) {
                    row.push(item.p.to_string());
                } else {
                    row.push("".to_string()); // Missing data
                }
            }

            wtr.write_record(&row)?;
        }

        wtr.flush()?;
        Ok(())
    }
}

impl Default for PolymarketScraper {
    fn default() -> Self {
        Self::new()
    }
}
