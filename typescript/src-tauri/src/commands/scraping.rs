/*!
 * Data scraping commands.
 */

use nglab::web::polymarket::{Frequency, PolymarketScraper};
use nglab::web::scraper::WebScraper;

/**
 * Scrapes historical market data from Polymarket for multiple outcome tokens.
 *
 * This command bridges the Tauri frontend to the `PolymarketScraper` in the core library.
 * It handles frequency parsing, date range conversion, and market resolution.
 *
 * @param market_source Optional market ID, slug, or URL to resolve.
 * @param token_ids List of specific token IDs to scrape if the source is not provided.
 * @param frequency Data sampling frequency (e.g., "1m", "1h", "Daily").
 * @param start_date Optional ISO-8601 start date for historical data.
 * @param end_date Optional ISO-8601 end date for historical data.
 * @param output_path File system path where the resulting CSV will be saved.
 */
#[tauri::command]
pub async fn scrape_polymarket(
    market_source: Option<String>,
    token_ids: Vec<String>,
    frequency: String,
    start_date: Option<String>,
    end_date: Option<String>,
    output_path: String,
) -> Result<(), String> {
    let freq = match frequency.as_str() {
        "1m" | "Minutely" => Frequency::Minutely,
        "1h" | "Hourly" => Frequency::Hourly,
        "1d" | "Daily" => Frequency::Daily,
        "1w" | "Weekly" => Frequency::Weekly,
        "30d" | "Monthly" => Frequency::Monthly,
        _ => return Err(format!("Invalid frequency: {}", frequency)),
    };

    let date_range = if let Some(start) = start_date {
        let start_dt = chrono::DateTime::parse_from_rfc3339(&start)
            .map_err(|e| format!("Invalid start date: {}", e))?
            .with_timezone(&chrono::Utc);
        let end_dt = if let Some(end) = end_date {
            chrono::DateTime::parse_from_rfc3339(&end)
                .map_err(|e| format!("Invalid end date: {}", e))?
                .with_timezone(&chrono::Utc)
        } else {
            chrono::Utc::now()
        };
        Some((start_dt, end_dt))
    } else {
        None
    };

    let mut scraper = PolymarketScraper::new().with_frequency(freq);

    if let Some((start, end)) = date_range {
        scraper = scraper.with_date_range(start, end);
    }

    if let Some(mut source) = market_source {
        if !source.trim().is_empty() {
            // If source passes, resolve full market first to get Names

            // Clean URL if needed
            if source.starts_with("http") {
                if let Ok(url) = url::Url::parse(&source) {
                    if let Some(segments) = url.path_segments() {
                        let segments_vec: Vec<&str> = segments.collect();
                        if let Some(last) = segments_vec.last() {
                            if !last.is_empty() {
                                source = last.to_string();
                            } else if segments_vec.len() > 1 {
                                source = segments_vec[segments_vec.len() - 2].to_string();
                            }
                        }
                    }
                }
            }

            scraper
                .resolve_market(&source)
                .await
                .map_err(|e| format!("Failed to resolve market source: {}", e))?;

            // If user selected specific tokens, filter them now
            if !token_ids.is_empty() {
                scraper = scraper.filter_options(token_ids);
            }
        } else {
            // Empty string source, fallback to legacy
            if !token_ids.is_empty() {
                scraper = scraper.with_options(token_ids);
            }
        }
    } else {
        // Legacy mode: Try to infer from first token_id if it looks like a URL/Slug
        if token_ids.is_empty() {
            return Err("No input provided".to_string());
        }

        let mut target_id = token_ids[0].clone();
        let mut is_url = false;

        if target_id.starts_with("http") || !target_id.chars().all(char::is_numeric) {
            is_url = true; // Heuristic
                           // ... clean url logic ...
            if target_id.starts_with("http") {
                if let Ok(url) = url::Url::parse(&target_id) {
                    if let Some(segments) = url.path_segments() {
                        let segments_vec: Vec<&str> = segments.collect();
                        if let Some(last) = segments_vec.last() {
                            if !last.is_empty() {
                                target_id = last.to_string();
                            } else if segments_vec.len() > 1 {
                                target_id = segments_vec[segments_vec.len() - 2].to_string();
                            }
                        }
                    }
                }
            }
        }

        // Try resolve
        if scraper.resolve_market(&target_id).await.is_ok() {
            // Good
        } else {
            // Failure
            if !is_url {
                // Maybe it was just a token ID?
                scraper = scraper.with_options(token_ids);
            } else {
                return Err(format!("Failed to resolve: {}", target_id));
            }
        }
    }

    scraper
        .download_csv(&output_path)
        .await
        .map_err(|e| format!("Scraping failed: {}", e))
}

/**
 * Resolves a Polymarket market identifier (slug, URL, or ID) into full metadata.
 *
 * @param input The market identifier to resolve.
 * @return Returns metadata including the market title and list of outcomes.
 */
#[tauri::command]
pub async fn resolve_polymarket_id(
    input: String,
) -> Result<nglab::web::polymarket::MarketMetadata, String> {
    let mut scraper = PolymarketScraper::new();

    // precise URL parsing to get the slug (Reuse logic)
    let mut target_id = input.clone();
    if target_id.starts_with("http") {
        if let Ok(url) = url::Url::parse(&target_id) {
            if let Some(segments) = url.path_segments() {
                let segments_vec: Vec<&str> = segments.collect();
                if let Some(last) = segments_vec.last() {
                    if !last.is_empty() {
                        target_id = last.to_string();
                    } else if segments_vec.len() > 1 {
                        target_id = segments_vec[segments_vec.len() - 2].to_string();
                    }
                }
            }
        }
    }

    scraper
        .resolve_market(&target_id)
        .await
        .map_err(|e| format!("Failed to resolve market: {}", e))?;

    Ok(scraper.get_metadata())
}
