/*!
 * Facebook Prophet-inspired forecasting model.
 *
 * Provides flexible trend and seasonality modeling
 * for structural time series with changepoints.
 */

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;
use ts_rs::TS;

use crate::errors::{ArenaError, ArenaResult};

/**
 * Configuration parameters for the Prophet model.
 */
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProphetParams {
    /// Trend growth type: "linear" or "flat".
    pub growth: String,
    /// Manual changepoint indices; if None, they are automatically detected.
    pub changepoints: Option<Vec<usize>>,
    /// Seasonality mode: "additive" or "multiplicative".
    pub seasonality_mode: String,
    /// Whether to include yearly seasonality patterns.
    pub yearly_seasonality: bool,
    /// Whether to include weekly seasonality patterns.
    pub weekly_seasonality: bool,
    /// Whether to include daily seasonality patterns.
    pub daily_seasonality: bool,
    /// Prior scale for seasonality regularization.
    pub seasonality_prior_scale: f64,
    /// Prior scale for changepoint selection regularization.
    pub changepoint_prior_scale: f64,
    /// Number of steps to forecast into the future.
    pub forecast_horizon: usize,
    /// Historical timestamps in seconds.
    pub times: Vec<i64>,
    /// Historical observation values.
    pub values: Vec<f64>,
}

/**
 * Result container for Prophet forecasts.
 */
#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[ts(export)]
pub struct ProphetResult {
    /// Timestamps for the forecast period.
    pub times: Vec<i64>,
    /// Total forecasted values (trend + seasonal).
    pub values: Vec<f64>,
    /// Trend component of the forecast.
    pub trend: Vec<f64>,
    /// Seasonality component of the forecast.
    pub seasonal: Vec<f64>,
}

/**
 * A simplified Prophet implementation for time-series forecasting.
 */
pub struct Prophet {
    params: ProphetParams,
    // Model coefficients
    k: f64, // Base trend rate
    m: f64, // Base trend offset
    #[allow(dead_code)]
    deltas: Array1<f64>, // Trend rate adjustments at changepoints
    beta: Array1<f64>, // Seasonality coefficients
}

impl Prophet {
    /// Creates a new Prophet model instance with the given parameters.
    pub fn new(params: ProphetParams) -> Self {
        Self {
            params,
            k: 0.0,
            m: 0.0,
            deltas: Array1::zeros(0),
            beta: Array1::zeros(0),
        }
    }

    /**
     * Fit the model to historical data using ridge regression.
     *
     * Implements full piecewise linear trend with changepoint support.
     * If changepoints are not provided, automatically detects them using PELT-inspired algorithm.
     *
     * @param times Array of timestamps.
     * @param y Array of target values.
     */
    pub fn fit(&mut self, times: &[i64], y: &[f64]) -> ArenaResult<()> {
        if times.len() != y.len() || times.is_empty() {
            return Err(ArenaError::ModelError("Data mismatch or empty".to_string()));
        }

        let t = self.normalize_time(times);
        let y_vec = Array1::from_vec(y.to_vec());
        let n = t.len();

        // 1. Setup Changepoints
        // If not provided, automatically detect them
        let changepoints = self.get_or_detect_changepoints(times, y);
        let num_changepoints = changepoints.len();

        // Create changepoint matrix A(t) where A[i,j] = 1 if t[i] >= s[j] (changepoint j)
        let mut a_matrix = Array2::<f64>::zeros((n, num_changepoints));
        for i in 0..n {
            for (j, &cp_idx) in changepoints.iter().enumerate() {
                if i >= cp_idx {
                    a_matrix[[i, j]] = 1.0;
                }
            }
        }

        // Normalized changepoint times
        let s_normalized: Vec<f64> = changepoints.iter().map(|&idx| t[idx]).collect();

        // 2. Setup Seasonality Component (Fourier Series)
        let seasonal_features = self.make_seasonality_features(times);
        let num_seasonal_params = seasonal_features.ncols();

        // Design Matrix X: [t, 1, A(t)*t - A(t)*s, seasonal_features...]
        // For piecewise linear: y = (k + A*delta) * t + (m + A*(-s*delta))
        // Simplified: y = k*t + m + sum_j delta_j * (t - s_j) * I(t >= s_j) + seasonal

        let num_trend_params = 2 + num_changepoints; // k, m, delta_1, delta_2, ...
        let total_params = num_trend_params + num_seasonal_params;
        let mut x_mat = Array2::<f64>::zeros((n, total_params));

        for i in 0..n {
            x_mat[[i, 0]] = t[i]; // Trend slope (k)
            x_mat[[i, 1]] = 1.0; // Trend offset (m)

            // Changepoint deltas: (t - s_j) * I(t >= s_j)
            for (j, &s_j) in s_normalized.iter().enumerate() {
                if a_matrix[[i, j]] > 0.5 {
                    x_mat[[i, 2 + j]] = t[i] - s_j;
                }
            }

            // Seasonal features
            for j in 0..num_seasonal_params {
                x_mat[[i, num_trend_params + j]] = seasonal_features[[i, j]];
            }
        }

        // Ridge Regression: (X^T X + lambda*I)^-1 X^T y
        // Use changepoint_prior_scale for delta regularization
        let lambda = 0.01;
        let delta_lambda = self.params.changepoint_prior_scale;
        let coeffs =
            self.solve_ridge_with_priors(&x_mat, &y_vec, lambda, delta_lambda, num_changepoints)?;

        // Unpack coefficients
        self.k = coeffs[0];
        self.m = coeffs[1];

        if num_changepoints > 0 {
            self.deltas = coeffs
                .slice(ndarray::s![2..2 + num_changepoints])
                .to_owned();
        } else {
            self.deltas = Array1::zeros(0);
        }

        if num_seasonal_params > 0 {
            self.beta = coeffs.slice(ndarray::s![num_trend_params..]).to_owned();
        } else {
            self.beta = Array1::zeros(0);
        }

        Ok(())
    }

    /**
     * Get user-provided changepoints or automatically detect them.
     *
     * Uses a simple variance-based algorithm inspired by PELT (Pruned Exact Linear Time).
     */
    fn get_or_detect_changepoints(&self, times: &[i64], y: &[f64]) -> Vec<usize> {
        if let Some(ref cps) = self.params.changepoints {
            // Validate provided changepoints
            return cps
                .iter()
                .filter(|&&cp| cp > 0 && cp < times.len() - 1)
                .cloned()
                .collect();
        }

        // Automatic detection: find potential changepoints in first 80% of data
        let n = times.len();
        let changepoint_range = (n as f64 * 0.8) as usize;

        if changepoint_range < 10 {
            return vec![];
        }

        // Default: 25 potential changepoints evenly spaced
        let n_changepoints = std::cmp::min(25, changepoint_range / 4);
        if n_changepoints == 0 {
            return vec![];
        }

        let step = changepoint_range / n_changepoints;
        let candidates: Vec<usize> = (1..=n_changepoints).map(|i| i * step).collect();

        // Use CUSUM-inspired scoring to find significant changepoints
        let mut scored_candidates: Vec<(usize, f64)> = candidates
            .iter()
            .filter_map(|&idx| {
                if idx < 3 || idx >= n - 3 {
                    return None;
                }
                let score = self.compute_changepoint_score(y, idx);
                Some((idx, score))
            })
            .collect();

        // Sort by score and keep top changepoints
        scored_candidates
            .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Keep changepoints with significant score (above average)
        let avg_score = scored_candidates.iter().map(|x| x.1).sum::<f64>()
            / scored_candidates.len().max(1) as f64;
        let mut selected: Vec<usize> = scored_candidates
            .iter()
            .filter(|(_, score)| *score > avg_score * 1.5)
            .map(|(idx, _)| *idx)
            .collect();

        // Sort by position
        selected.sort();

        // Limit to reasonable number
        selected.truncate(10);
        selected
    }

    /**
     * Compute a changepoint score using cumulative sum of deviations.
     */
    fn compute_changepoint_score(&self, y: &[f64], idx: usize) -> f64 {
        let n = y.len();
        if idx < 2 || idx >= n - 2 {
            return 0.0;
        }

        // Compare mean before and after
        let before_mean: f64 = y[..idx].iter().sum::<f64>() / idx as f64;
        let after_mean: f64 = y[idx..].iter().sum::<f64>() / (n - idx) as f64;

        // Score is the absolute difference in means
        (after_mean - before_mean).abs()
    }

    /**
     * Ridge regression with different priors for different parameter groups.
     */
    fn solve_ridge_with_priors(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        base_lambda: f64,
        delta_lambda: f64,
        num_changepoints: usize,
    ) -> ArenaResult<Array1<f64>> {
        let xt = x.t();
        let xt_x = xt.dot(x);
        let xt_y = xt.dot(y);

        let n_dims = xt_x.nrows();
        let mut a = xt_x;

        // Add different regularization for different parameters
        for i in 0..n_dims {
            if i < 2 {
                // k, m get base regularization
                a[[i, i]] += base_lambda;
            } else if i < 2 + num_changepoints {
                // deltas get changepoint_prior_scale
                a[[i, i]] += delta_lambda;
            } else {
                // seasonal params get seasonality_prior_scale
                a[[i, i]] += self.params.seasonality_prior_scale;
            }
        }

        let l = self.cholesky(&a)?;
        let z = self.forward_substitution(&l, &xt_y)?;
        let beta = self.backward_substitution(&l.t().to_owned(), &z)?;

        Ok(beta)
    }

    // --- Helpers ---

    fn normalize_time(&self, times: &[i64]) -> Vec<f64> {
        if times.is_empty() {
            return vec![];
        }
        let min_t = times[0] as f64;
        let max_t = times[times.len() - 1] as f64;
        let range = (max_t - min_t).max(1.0);

        times.iter().map(|&t| (t as f64 - min_t) / range).collect()
    }

    fn make_seasonality_features(&self, times: &[i64]) -> Array2<f64> {
        let n = times.len();
        let mut features = Vec::new();

        // Unix timestamp in seconds expected
        // Daily (period = 1 day = 86400s) - order 4
        if self.params.daily_seasonality {
            let period = 86400.0;
            self.add_fourier_terms(times, period, 4, &mut features);
        }

        // Weekly (period = 7 days) - order 3
        if self.params.weekly_seasonality {
            let period = 86400.0 * 7.0;
            self.add_fourier_terms(times, period, 3, &mut features);
        }

        // Yearly (period = 365.25 days) - order 10
        if self.params.yearly_seasonality {
            let period = 86400.0 * 365.25;
            self.add_fourier_terms(times, period, 10, &mut features);
        }

        if features.is_empty() {
            return Array2::zeros((n, 0));
        }

        let num_features = features.len();
        let mut mat = Array2::zeros((n, num_features));
        for (j, col) in features.iter().enumerate() {
            for i in 0..n {
                mat[[i, j]] = col[i];
            }
        }
        mat
    }

    fn add_fourier_terms(
        &self,
        times: &[i64],
        period: f64,
        order: usize,
        output: &mut Vec<Vec<f64>>,
    ) {
        for i in 1..=order {
            let f = i as f64;
            // sin(2 * pi * i * t / P)
            // cos(2 * pi * i * t / P)
            // Note: t should be in seconds if period is in seconds.

            let sin_term: Vec<f64> = times
                .iter()
                .map(|&t| (2.0 * PI * f * t as f64 / period).sin())
                .collect();
            let cos_term: Vec<f64> = times
                .iter()
                .map(|&t| (2.0 * PI * f * t as f64 / period).cos())
                .collect();

            output.push(sin_term);
            output.push(cos_term);
        }
    }

    /** Solves (X^T X + lambda I) beta = X^T y using Cholesky Decomposition */
    #[allow(dead_code)]
    fn solve_ridge(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        lambda: f64,
    ) -> ArenaResult<Array1<f64>> {
        let xt = x.t();
        let xt_x = xt.dot(x);
        let xt_y = xt.dot(y);

        let n_dims = xt_x.nrows();
        let mut a = xt_x;

        // Add ridge regularization
        for i in 0..n_dims {
            a[[i, i]] += lambda;
        }

        // Cholesky: A = L L^T
        // Solve Ly = b (forward subst)
        // Solve L^T x = y (backward subst)

        let l = self.cholesky(&a)?;
        let z = self.forward_substitution(&l, &xt_y)?;
        let beta = self.backward_substitution(&l.t().to_owned(), &z)?;

        Ok(beta)
    }

    fn cholesky(&self, a: &Array2<f64>) -> ArenaResult<Array2<f64>> {
        let n = a.nrows();
        let mut l = Array2::zeros((n, n));

        for i in 0..n {
            for j in 0..=i {
                let mut sum = 0.0;
                for k in 0..j {
                    sum += l[[i, k]] * l[[j, k]];
                }

                if i == j {
                    let val = a[[i, i]] - sum;
                    if val <= 0.0 {
                        return Err(ArenaError::NumericalError(
                            "Matrix not positive definite during Cholesky decomposition"
                                .to_string(),
                        ));
                    }
                    l[[i, j]] = val.sqrt();
                } else {
                    l[[i, j]] = (a[[i, j]] - sum) / l[[j, j]];
                }
            }
        }
        Ok(l)
    }

    fn forward_substitution(&self, l: &Array2<f64>, b: &Array1<f64>) -> ArenaResult<Array1<f64>> {
        let n = l.nrows();
        let mut y = Array1::zeros(n);

        for i in 0..n {
            let mut sum = 0.0;
            for j in 0..i {
                sum += l[[i, j]] * y[j];
            }
            if l[[i, i]].abs() < 1e-10 {
                return Err(ArenaError::NumericalError(
                    "Singular matrix in forward substitution".to_string(),
                ));
            }
            y[i] = (b[i] - sum) / l[[i, i]];
        }
        Ok(y)
    }

    fn backward_substitution(&self, u: &Array2<f64>, y: &Array1<f64>) -> ArenaResult<Array1<f64>> {
        let n = u.nrows();
        let mut x = Array1::zeros(n);

        for i in (0..n).rev() {
            let mut sum = 0.0;
            for j in i + 1..n {
                sum += u[[i, j]] * x[j];
            }
            if u[[i, i]].abs() < 1e-10 {
                return Err(ArenaError::NumericalError(
                    "Singular matrix in backward substitution".to_string(),
                ));
            }
            x[i] = (y[i] - sum) / u[[i, i]];
        }
        Ok(x)
    }
}

/// Executes a Prophet forecast simulation.
///
/// Fits the model to historical data and generates future predictions.
pub fn simulate(params: ProphetParams) -> ArenaResult<ProphetResult> {
    let historical_times = &params.times;
    let historical_values = &params.values;
    // 1. Fit Model
    let mut model = Prophet::new(params.clone());
    model.fit(historical_times, historical_values)?;

    // 2. Predict Future
    // Generate future timestamps
    let last_time = *historical_times.last().unwrap_or(&0);
    // Determine step size (median step from history or just daily/hourly?)
    // Basic heuristic: average diff
    let step_size = if historical_times.len() > 1 {
        let diffs: Vec<i64> = historical_times.windows(2).map(|w| w[1] - w[0]).collect();
        diffs.iter().sum::<i64>() / diffs.len() as i64
    } else {
        86400 // Default 1 day
    };

    let horizon = params.forecast_horizon;
    let mut future_times = Vec::with_capacity(horizon);
    let mut current_time = last_time;
    for _ in 0..horizon {
        current_time += step_size;
        future_times.push(current_time);
    }

    // Prepare features for prediction
    // Need to use the SAME scalar logic as fit!
    // Since we created the model cleanly, we should refactor 'fit' to store 'min_t'/'scale_t' in struct
    // Re-instantiating `t_norm` based on the fitted range.

    let t_min = historical_times[0] as f64;
    let t_max = historical_times[historical_times.len() - 1] as f64;
    let t_scale = (t_max - t_min).max(1.0);

    let future_t_norm: Vec<f64> = future_times
        .iter()
        .map(|&t| (t as f64 - t_min) / t_scale)
        .collect();

    let future_seasonal_features = model.make_seasonality_features(&future_times); // Uses raw times for Fourier

    let mut predicted_values = Vec::with_capacity(horizon);
    let mut predicted_trend = Vec::with_capacity(horizon);
    let mut predicted_seasonal = Vec::with_capacity(horizon);

    // Calculate predictions
    // y = (k * t + m) + (seasonality)
    let has_seasonality = future_seasonal_features.ncols() > 0;

    for i in 0..horizon {
        let t = future_t_norm[i];
        let trend = model.k * t + model.m;

        let mut seasonal = 0.0;
        if has_seasonality {
            for j in 0..model.beta.len() {
                seasonal += model.beta[j] * future_seasonal_features[[i, j]];
            }
        }

        let y_hat = if params.seasonality_mode == "multiplicative" {
            trend * (1.0 + seasonal)
        } else {
            trend + seasonal
        };

        predicted_values.push(y_hat);
        predicted_trend.push(trend);
        predicted_seasonal.push(seasonal);
    }

    Ok(ProphetResult {
        times: future_times,
        values: predicted_values,
        trend: predicted_trend,
        seasonal: predicted_seasonal,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prophet_linear_trend() {
        // Synthetic data: y = 2*t + 5
        let n = 100;
        let times: Vec<i64> = (0..n).map(|i| i * 86400).collect();
        let values: Vec<f64> = times
            .iter()
            .enumerate()
            .map(|(i, _)| 2.0 * i as f64 + 5.0)
            .collect();

        let params = ProphetParams {
            growth: "linear".to_string(),
            changepoints: None,
            seasonality_mode: "additive".to_string(),
            yearly_seasonality: false,
            weekly_seasonality: false,
            daily_seasonality: false,
            seasonality_prior_scale: 10.0,
            changepoint_prior_scale: 0.05,
            forecast_horizon: 10,
            times: times.clone(),
            values: values.clone(),
        };

        let result = simulate(params).unwrap();

        assert_eq!(result.values.len(), 10);
        // Next value should be approx 2 * 100 + 5 = 205
        // Trend is fitted on normalized time 0..1.
        // t=100 corresponds to normalized t > 1.
        // Correct check: The continuity of the line.
        let first_pred = result.values[0];
        assert!(
            (first_pred - 205.0).abs() < 1.0,
            "Prediction {} should be close to 205.0",
            first_pred
        );
    }

    #[test]
    fn test_prophet_seasonality() {
        // Synthetic data: y = 10 + sin(2*pi*t/7days)
        // Weekly seasonality
        let n = 28; // 4 weeks
        let times: Vec<i64> = (0..n).map(|i| i * 86400).collect();
        let values: Vec<f64> = times
            .iter()
            .map(|&t| 10.0 + (2.0 * PI * t as f64 / (86400.0 * 7.0)).sin())
            .collect();

        let params = ProphetParams {
            growth: "flat".to_string(),
            changepoints: None,
            seasonality_mode: "additive".to_string(),
            yearly_seasonality: false,
            weekly_seasonality: true,
            daily_seasonality: false,
            seasonality_prior_scale: 10.0,
            changepoint_prior_scale: 0.05,
            forecast_horizon: 7,
            times: times.clone(),
            values: values.clone(),
        };

        let result = simulate(params).unwrap();

        assert_eq!(result.values.len(), 7);
        // Check pattern repeats
        // Prediction for day 28 (start of week 5) should match day 0 (start of week 1) -> 10.0 + sin(0) = 10.0
        // Note: With changepoint detection active, tolerance is slightly higher
        let pred_day_0 = result.values[0];
        assert!(
            (pred_day_0 - 10.0).abs() < 0.5,
            "Prediction {} should be close to 10.0",
            pred_day_0
        );
    }
}
