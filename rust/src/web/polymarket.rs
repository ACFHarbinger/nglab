use crate::errors::{ArenaError, ArenaResult};
use crate::web::scraper::WebScraper;
use chrono::{DateTime, Utc};
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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

#[derive(Serialize, Clone, Debug)]
pub struct OutcomeInfo {
    pub id: String,
    pub name: String,
}

#[derive(Serialize, Clone, Debug)]
pub struct MarketMetadata {
    pub title: String,
    pub outcomes: Vec<OutcomeInfo>,
}

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

    pub fn resolve_market(&mut self, market_id: &str) -> ArenaResult<()> {
        let url = format!("https://gamma-api.polymarket.com/markets/{}", market_id);

        // Try to fetch as a single market first
        let response = self.client.get(&url).send()?.error_for_status();

        match response {
            Ok(resp) => {
                let market: MarketResponse = resp.json()?;
                self.process_market(market);
                Ok(())
            }
            Err(_) => {
                // If failed, maybe it's an Event ID? Try events endpoint
                let url = format!("https://gamma-api.polymarket.com/events?id={}", market_id);
                if let Ok(response) = self.client.get(&url).send()?.error_for_status() {
                    let events: Vec<EventResponse> = response.json()?;
                    self.process_events(events)?;
                    Ok(())
                } else {
                    // Finally, try as a SLUG (Event first, then Market)
                    let url = format!("https://gamma-api.polymarket.com/events?slug={}", market_id);
                    if let Ok(response) = self.client.get(&url).send()?.error_for_status() {
                        let events: Vec<EventResponse> = response.json()?;
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
                    if let Ok(response) = self.client.get(&url).send()?.error_for_status() {
                        let markets: Vec<MarketResponse> = response.json()?;
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

    fn fetch_history(&self, token_id: &str) -> ArenaResult<Vec<HistoryItem>> {
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
            .send()?
            .error_for_status()?;

        let data: HistoryResponse = response.json()?;

        Ok(data.history)
    }
}

#[derive(Deserialize, Debug, Clone)]
struct MarketResponse {
    // id: String,
    question: String,
    outcomes: String, // Stringified JSON array: "[\"Yes\", \"No\"]"
    #[serde(rename = "clobTokenIds")]
    clob_token_ids: String, // Stringified JSON array
}

#[derive(Deserialize, Debug)]
struct EventResponse {
    // id: String,
    title: String,
    markets: Vec<MarketResponse>,
}

impl WebScraper for PolymarketScraper {
    fn download_csv(&self, output_path: &str) -> ArenaResult<()> {
        if self.token_ids.is_empty() {
            return Err(ArenaError::InvalidOrder(
                "No options (token IDs) selected".to_string(),
            ));
        }

        // Fetch data for all options
        let mut all_histories = HashMap::new();
        let mut all_timestamps = Vec::new();

        for token_id in &self.token_ids {
            let history = self.fetch_history(token_id)?;
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
