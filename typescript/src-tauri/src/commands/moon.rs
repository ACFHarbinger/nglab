/*!
 * Time-series forecasting commands (Project Moon).
 */

/**
 * Predicts future prices using an ARIMA(p,d,q) model.
 *
 * @param params Parameters for the ARIMA simulation and integration.
 * @return Returns the prediction result containing the simulated price path.
 */
/**
 * Predicts future prices using an ARIMA(p,d,q) model with fitting.
 */
#[tauri::command]
pub async fn run_arima(
    data: Vec<f64>,
    p: usize,
    d: usize,
    q: usize,
    steps: usize,
) -> Result<nglab::moon::arima::ArimaResult, String> {
    tauri::async_runtime::spawn_blocking(move || {
        nglab::moon::arima::fit_and_simulate(data, p, d, q, steps)
    })
    .await
    .map_err(|e| format!("Simulation task failed: {}", e))?
    .map_err(|e| e.to_string())
}

/**
 * Predicts future volatility and returns using a GARCH(1,1) model.
 */
#[tauri::command]
pub async fn run_garch(
    params: nglab::moon::garch::GarchParams,
) -> Result<nglab::moon::garch::GarchResult, String> {
    tauri::async_runtime::spawn_blocking(move || nglab::moon::garch::simulate(params))
        .await
        .map_err(|e| format!("Simulation task failed: {}", e))?
        .map_err(|e| e.to_string())
}

/**
 * Predicts future values using triple exponential smoothing (Holt-Winters).
 */
#[tauri::command]
pub async fn run_holt_winters(
    params: nglab::moon::es::HoltWintersParams,
) -> Result<nglab::moon::es::HoltWintersResult, String> {
    tauri::async_runtime::spawn_blocking(move || nglab::moon::es::simulate(params))
        .await
        .map_err(|e| format!("Simulation task failed: {}", e))?
        .map_err(|e| e.to_string())
}

/**
 * Predicts future values using the Prophet-lite statistical model.
 */
#[tauri::command]
pub async fn run_prophet(
    params: nglab::moon::prophet::ProphetParams,
) -> Result<nglab::moon::prophet::ProphetResult, String> {
    tauri::async_runtime::spawn_blocking(move || nglab::moon::prophet::simulate(params))
        .await
        .map_err(|e| format!("Simulation task failed: {}", e))?
        .map_err(|e| e.to_string())
}
