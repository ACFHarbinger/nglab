/*!
 * Time-series forecasting commands (Project Moon).
 */

/**
 * Predicts future prices using an ARIMA(p,d,q) model.
 *
 * @param params Parameters for the ARIMA simulation and integration.
 * @return Returns the prediction result containing the simulated price path.
 */
#[tauri::command]
pub async fn predict_arima(
    params: nglab::moon::arima::ArimaParams,
) -> Result<nglab::moon::arima::ArimaResult, String> {
    tauri::async_runtime::spawn_blocking(move || nglab::moon::arima::simulate(params))
        .await
        .map_err(|e| format!("Simulation task failed: {}", e))?
}

/**
 * Predicts future volatility and returns using a GARCH(1,1) model.
 *
 * @param params Input parameters for the GARCH simulation.
 * @return Returns the prediction result with simulated paths.
 */
#[tauri::command]
pub async fn predict_garch(
    params: nglab::moon::garch::GarchParams,
) -> Result<nglab::moon::garch::GarchResult, String> {
    tauri::async_runtime::spawn_blocking(move || nglab::moon::garch::simulate(params))
        .await
        .map_err(|e| format!("Simulation task failed: {}", e))?
}

/**
 * Predicts future values using triple exponential smoothing (Holt-Winters).
 *
 * @param params Seasonality, trend, and smoothing parameters.
 * @return Returns the forecast result.
 */
#[tauri::command]
pub async fn predict_holt_winters(
    params: nglab::moon::es::HoltWintersParams,
) -> Result<nglab::moon::es::HoltWintersResult, String> {
    tauri::async_runtime::spawn_blocking(move || nglab::moon::es::simulate(params))
        .await
        .map_err(|e| format!("Simulation task failed: {}", e))?
}

/**
 * Predicts future values using the Prophet-lite statistical model.
 *
 * @param params trend, seasonal, and changepoint parameters.
 * @return Returns the forecast result.
 */
#[tauri::command]
pub async fn predict_prophet(
    params: nglab::moon::prophet::ProphetParams,
) -> Result<nglab::moon::prophet::ProphetResult, String> {
    tauri::async_runtime::spawn_blocking(move || nglab::moon::prophet::simulate(params))
        .await
        .map_err(|e| format!("Simulation task failed: {}", e))?
}
