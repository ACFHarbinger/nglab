use crate::error::{ArenaError, ArenaResult};
use crate::web::scraper::WebScraper;
use chrono::{DateTime, Utc};
use reqwest::blocking::Client;
use serde::Deserialize;
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

pub struct PolymarketScraper {
    client: Client,
    token_ids: Vec<String>,
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

    fn fetch_history(&self, token_id: &str) -> ArenaResult<Vec<HistoryItem>> {
        let (interval, _fidelity) = self.frequency.to_interval();

        let mut params = vec![
            ("market", token_id),
            ("interval", interval),
            ("fidelity", "1"), // Default fidelity
        ];

        let start_ts_str;
        let end_ts_str;

        if let Some(start) = self.start_date {
            start_ts_str = start.timestamp().to_string();
            params.push(("startTs", &start_ts_str));
        }

        if let Some(end) = self.end_date {
            end_ts_str = end.timestamp().to_string();
            params.push(("endTs", &end_ts_str));
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

        let mut headers = vec!["Date".to_string(), "Timestamp".to_string()];
        for token_id in &self.token_ids {
            headers.push(format!("Price_{}", token_id));
        }
        wtr.write_record(&headers)?;

        for ts in all_timestamps {
            let mut row = Vec::new();

            // Date string
            if let Some(datetime) = DateTime::from_timestamp(ts as i64, 0) {
                row.push(datetime.to_rfc3339());
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
