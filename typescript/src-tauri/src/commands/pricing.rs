/*!
 * Option pricing commands.
 */

/**
 * Prices an option using the Rough Bergomi (rBergomi) stochastic volatility model.
 *
 * @param params Parameters for the rBergomi simulation.
 * @return Returns the simulation result containing the simulated price path.
 */
#[tauri::command]
pub async fn pricing_rbergomi(
    params: nglab::models::rough_bergomi::RBergomiParams,
) -> Result<nglab::models::rough_bergomi::RBergomiResult, String> {
    tauri::async_runtime::spawn_blocking(move || nglab::models::rough_bergomi::simulate(params))
        .await
        .map_err(|e| format!("Simulation task failed: {}", e))?
}

/**
 * Prices a European option using the classic Black-Scholes-Merton model.
 *
 * @param params Input parameters (stock price, strike, volatility, time to maturity, etc.).
 * @return Returns the theoretical price and Greeks.
 */
#[tauri::command]
pub async fn pricing_black_scholes(
    params: nglab::models::black_scholes::BlackScholesParams,
) -> Result<nglab::models::black_scholes::BlackScholesResult, String> {
    tauri::async_runtime::spawn_blocking(move || nglab::models::black_scholes::price(params))
        .await
        .map_err(|e| format!("Pricing task failed: {}", e))
}

/**
 * Calculates credit risk metrics, such as Credit Valuation Adjustment (CVA).
 *
 * @param params Exposure and counterparty default probability parameters.
 * @return Returns the risk evaluation results.
 */
#[tauri::command]
pub async fn pricing_credit_risk(
    params: nglab::models::credit_risk::CreditRiskParams,
) -> Result<nglab::models::credit_risk::CreditRiskResult, String> {
    tauri::async_runtime::spawn_blocking(move || nglab::models::credit_risk::price(params))
        .await
        .map_err(|e| format!("Pricing task failed: {}", e))?
}

/**
 * Prices an option using the Rough Heston stochastic volatility model.
 *
 * @param params Fractional integration and volatility clustering parameters.
 * @return Returns the simulated price and variance paths.
 */
#[tauri::command]
pub async fn pricing_rough_heston(
    params: nglab::models::rough_heston::RoughHestonParams,
) -> Result<nglab::models::rough_heston::RoughHestonResult, String> {
    tauri::async_runtime::spawn_blocking(move || nglab::models::rough_heston::simulate(params))
        .await
        .map_err(|e| format!("Simulation task failed: {}", e))?
}
