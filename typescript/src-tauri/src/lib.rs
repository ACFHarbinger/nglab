//! Tauri backend logic for the NGLab dashboard.
//!
//! This module coordinates the interaction between the Rust simulation
//! arena and the React frontend via Tauri commands and events.

use nglab::simulation::gym::TradingEnv;
use nglab::simulation::orderbook::OrderBook;
use nglab::web::polymarket::{Frequency, PolymarketScraper};
use nglab::web::scraper::WebScraper;
use std::sync::Mutex;
use tauri::{Emitter, Manager, State};
use tokio::time::{sleep, Duration};

/** Shared application state for the Tauri backend. */
struct ArenaState {
    env: Mutex<TradingEnv>,
    running: Mutex<bool>,
}

/** Real-time update event emitted to the frontend. */
#[derive(serde::Serialize, Clone)]
struct ArenaUpdate {
    step: u64,
    price: f64,
    portfolio_value: f64,
    orderbook: OrderBook,
}

/**
 * Command to start the simulation loop in the background.
 */
#[tauri::command]
fn start_simulation(state: State<ArenaState>, app: tauri::AppHandle) {
    let mut running = state.running.lock().unwrap();
    if *running {
        return;
    }
    *running = true;

    tauri::async_runtime::spawn(async move {
        let state = app.state::<ArenaState>();
        loop {
            // Check if we should keep running
            {
                let running = state.running.lock().unwrap();
                if !*running {
                    break;
                }
            }

            // Perform simulation step
            let update = {
                let mut env = state.env.lock().unwrap();
                // 0 = Hold action
                let (_, _, _, _, step_info) = env.step_rs(0);

                let orderbook = env.orderbook().clone();
                let price = orderbook.mid_price().unwrap_or(0.0);

                ArenaUpdate {
                    step: step_info.total_steps,
                    price,
                    portfolio_value: step_info.portfolio_value,
                    orderbook,
                }
            };

            // Emit event to frontend
            let _ = app.emit("arena-update", &update);

            sleep(Duration::from_millis(100)).await;
        }
    });
}

/**
 * Command to stop the active simulation loop.
 */
#[tauri::command]
fn stop_simulation(state: State<ArenaState>) {
    let mut running = state.running.lock().unwrap();
    *running = false;
}

/**
 * Command to scrape Polymarket data for multiple tokens.
 */
#[tauri::command]
async fn scrape_polymarket(
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

    tauri::async_runtime::spawn_blocking(move || {
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
            if scraper.resolve_market(&target_id).is_ok() {
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
            .map_err(|e| format!("Scraping failed: {}", e))
    })
    .await
    .map_err(|e| format!("Task failed: {}", e))?
}

/**
 * Command to resolve a Polymarket market SLUG or URL into metadata.
 */
#[tauri::command]
async fn resolve_polymarket_id(
    input: String,
) -> Result<nglab::web::polymarket::MarketMetadata, String> {
    tauri::async_runtime::spawn_blocking(move || {
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
            .map_err(|e| format!("Failed to resolve market: {}", e))?;

        Ok(scraper.get_metadata())
    })
    .await
    .map_err(|e| format!("Task failed: {}", e))?
}

/**
 * Command to price options using the Rough Bergomi model.
 */
#[tauri::command]
async fn pricing_rbergomi(
    params: nglab::models::rough_bergomi::RBergomiParams,
) -> Result<nglab::models::rough_bergomi::RBergomiResult, String> {
    tauri::async_runtime::spawn_blocking(move || nglab::models::rough_bergomi::simulate(params))
        .await
        .map_err(|e| format!("Simulation task failed: {}", e))?
}

/**
 * Command to price options using the Black-Scholes model.
 */
#[tauri::command]
async fn pricing_black_scholes(
    params: nglab::models::black_scholes::BlackScholesParams,
) -> Result<nglab::models::black_scholes::BlackScholesResult, String> {
    tauri::async_runtime::spawn_blocking(move || nglab::models::black_scholes::price(params))
        .await
        .map_err(|e| format!("Pricing task failed: {}", e))
}

/**
 * Command to calculate Credit Valuation Adjustment (CVA).
 */
#[tauri::command]
async fn pricing_credit_risk(
    params: nglab::models::credit_risk::CreditRiskParams,
) -> Result<nglab::models::credit_risk::CreditRiskResult, String> {
    tauri::async_runtime::spawn_blocking(move || nglab::models::credit_risk::price(params))
        .await
        .map_err(|e| format!("Pricing task failed: {}", e))?
}

/**
 * Command to price options using the Rough Heston model.
 */
#[tauri::command]
async fn pricing_rough_heston(
    params: nglab::models::rough_heston::RoughHestonParams,
) -> Result<nglab::models::rough_heston::RoughHestonResult, String> {
    tauri::async_runtime::spawn_blocking(move || nglab::models::rough_heston::simulate(params))
        .await
        .map_err(|e| format!("Simulation task failed: {}", e))?
}

/**
 * Command to predict future prices using ARIMA.
 */
#[tauri::command]
async fn predict_arima(
    params: nglab::moon::arima::ArimaParams,
) -> Result<nglab::moon::arima::ArimaResult, String> {
    tauri::async_runtime::spawn_blocking(move || nglab::moon::arima::simulate(params))
        .await
        .map_err(|e| format!("Simulation task failed: {}", e))?
}

/**
 * Command to predict future prices using GARCH.
 */
#[tauri::command]
async fn predict_garch(
    params: nglab::moon::garch::GarchParams,
) -> Result<nglab::moon::garch::GarchResult, String> {
    tauri::async_runtime::spawn_blocking(move || nglab::moon::garch::simulate(params))
        .await
        .map_err(|e| format!("Simulation task failed: {}", e))?
}

/**
 * Command to predict future prices using Holt-Winters (Exponential Smoothing).
 */
#[tauri::command]
async fn predict_holt_winters(
    params: nglab::moon::es::HoltWintersParams,
) -> Result<nglab::moon::es::HoltWintersResult, String> {
    tauri::async_runtime::spawn_blocking(move || nglab::moon::es::simulate(params))
        .await
        .map_err(|e| format!("Simulation task failed: {}", e))?
}

/**
 * Command to predict future prices using the Prophet model.
 */
#[tauri::command]
async fn predict_prophet(
    params: nglab::moon::prophet::ProphetParams,
) -> Result<nglab::moon::prophet::ProphetResult, String> {
    tauri::async_runtime::spawn_blocking(move || nglab::moon::prophet::simulate(params))
        .await
        .map_err(|e| format!("Simulation task failed: {}", e))?
}

/**
 * Entry point for the Tauri application.
 */
#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    // Initialize TradingEnv with default parameters
    let env = TradingEnv::new(10000.0, 0.001, 30, 1000, false);

    let state = ArenaState {
        env: Mutex::new(env),
        running: Mutex::new(false),
    };

    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .manage(state)
        .invoke_handler(tauri::generate_handler![
            start_simulation,
            stop_simulation,
            scrape_polymarket,
            resolve_polymarket_id,
            pricing_rbergomi,
            pricing_black_scholes,
            pricing_credit_risk,
            pricing_rough_heston,
            predict_arima,
            predict_garch,
            predict_holt_winters,
            predict_prophet
        ])
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|app_handle, event| match event {
            tauri::RunEvent::ExitRequested { .. } => {
                if let Ok(mut running) = app_handle.state::<ArenaState>().running.lock() {
                    *running = false;
                }
            }
            _ => {}
        });
}
