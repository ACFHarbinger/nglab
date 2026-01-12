//! Facebook Prophet-inspired forecasting model.
//!
//! Provides flexible trend and seasonality modeling
//! for structural time series with changepoints.

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProphetParams {
    pub growth: String,                   // "linear" or "flat"
    pub changepoints: Option<Vec<usize>>, // Indices of changepoints (optional)
    pub seasonality_mode: String,         // "additive" or "multiplicative"
    pub yearly_seasonality: bool,
    pub weekly_seasonality: bool,
    pub daily_seasonality: bool,
    pub seasonality_prior_scale: f64,
    pub changepoint_prior_scale: f64,
    pub forecast_horizon: usize,
    pub times: Vec<i64>,
    pub values: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProphetResult {
    pub times: Vec<i64>,
    pub values: Vec<f64>,
    pub trend: Vec<f64>,
    pub seasonal: Vec<f64>,
}

/** A simplified Prophet implementation in pure Rust. */
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
     * Fit the model to historical data.
     * `times` should be timestamps (e.g., seconds/milliseconds).
     * `y` is the target time series.
     */
    pub fn fit(&mut self, times: &[i64], y: &[f64]) -> Result<(), String> {
        if times.len() != y.len() || times.is_empty() {
            return Err("Data mismatch or empty".to_string());
        }

        let t = self.normalize_time(times);
        let y_vec = Array1::from_vec(y.to_vec());
        let n = t.len();

        // 1. Setup Trend Component
        // For simple linear trend: y = (k + A(t)delta) * t + (m + A(t)(-t_changes)delta)
        // Check params for changepoints. If none, just simple linear: y = kt + m
        // Implemented: Simple Linear Trend (Global) for now to start "Prophet-Lite"
        // TODO: Add changepoints support

        // 2. Setup Seasonality Component (Fourier Series)
        let seasonal_features = self.make_seasonality_features(times);
        let num_seasonal_params = seasonal_features.ncols();

        // Design Matrix X: [t, 1, seasonal_features...]
        // Note: For simple linear trend, we just need column `t` and column `1` (bias).

        let mut x_mat = Array2::<f64>::zeros((n, 2 + num_seasonal_params));

        for i in 0..n {
            x_mat[[i, 0]] = t[i]; // Trend feature
            x_mat[[i, 1]] = 1.0; // Bias/Offset feature

            for j in 0..num_seasonal_params {
                x_mat[[i, 2 + j]] = seasonal_features[[i, j]];
            }
        }

        // Ridge Regression: (X^T X + lambda*I)^-1 X^T y
        // lambda (regularization) depends on priors.
        // For simplicity in this Lite version, we use a small fixed lambda for stability
        // or derive from params.
        let lambda = 0.01;
        let coeffs = self.solve_ridge(&x_mat, &y_vec, lambda)?;

        // Unpack coefficients
        self.k = coeffs[0];
        self.m = coeffs[1];

        if num_seasonal_params > 0 {
            self.beta = coeffs.slice(ndarray::s![2..]).to_owned();
        } else {
            self.beta = Array1::zeros(0);
        }

        Ok(())
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
    fn solve_ridge(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        lambda: f64,
    ) -> Result<Array1<f64>, String> {
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

    fn cholesky(&self, a: &Array2<f64>) -> Result<Array2<f64>, String> {
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
                        return Err("Matrix not positive definite".to_string());
                    }
                    l[[i, j]] = val.sqrt();
                } else {
                    l[[i, j]] = (a[[i, j]] - sum) / l[[j, j]];
                }
            }
        }
        Ok(l)
    }

    fn forward_substitution(
        &self,
        l: &Array2<f64>,
        b: &Array1<f64>,
    ) -> Result<Array1<f64>, String> {
        let n = l.nrows();
        let mut y = Array1::zeros(n);

        for i in 0..n {
            let mut sum = 0.0;
            for j in 0..i {
                sum += l[[i, j]] * y[j];
            }
            if l[[i, i]].abs() < 1e-10 {
                return Err("Singular matrix in forward substitution".to_string());
            }
            y[i] = (b[i] - sum) / l[[i, i]];
        }
        Ok(y)
    }

    fn backward_substitution(
        &self,
        u: &Array2<f64>,
        y: &Array1<f64>,
    ) -> Result<Array1<f64>, String> {
        let n = u.nrows();
        let mut x = Array1::zeros(n);

        for i in (0..n).rev() {
            let mut sum = 0.0;
            for j in i + 1..n {
                sum += u[[i, j]] * x[j];
            }
            if u[[i, i]].abs() < 1e-10 {
                return Err("Singular matrix in backward substitution".to_string());
            }
            x[i] = (y[i] - sum) / u[[i, i]];
        }
        Ok(x)
    }
}

/**
 * Simulate/Forecast using the Prophet model.
 */
pub fn simulate(params: ProphetParams) -> Result<ProphetResult, String> {
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
        let pred_day_0 = result.values[0];
        assert!(
            (pred_day_0 - 10.0).abs() < 0.1,
            "Prediction {} should be close to 10.0",
            pred_day_0
        );
    }
}
