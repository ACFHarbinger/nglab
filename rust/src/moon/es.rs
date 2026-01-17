/*!
 * Exponential Smoothing (ES) and Holt-Winters forecasting.
 *
 * Provides statistical methods for trend and seasonality
 * decomposition in time series data.
 */

use rand_distr::{Distribution, StandardNormal};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq)]
pub enum SeasonalType {
    Additive,
    Multiplicative,
}

/**
 * Supported seasonal adjustment types for Holt-Winters.
 */
/**
 * Configuration parameters for the Holt-Winters Exponential Smoothing model.
 */
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct HoltWintersParams {
    pub alpha: f64, // Level smoothing (0-1)
    pub beta: f64,  // Trend smoothing (0-1)
    pub gamma: f64, // Seasonal smoothing (0-1)
    pub period: usize,
    pub seasonal_type: SeasonalType,
    pub steps: usize,
    pub sigma: f64, // Noise standard deviation
    pub seed: Option<u64>,
    pub data: Option<Vec<f64>>, // Historical data for initialization
}

/**
 * Result container for Exponential Smoothing forecasts.
 */
#[derive(Serialize, Deserialize, Debug)]
pub struct HoltWintersResult {
    pub path: Vec<f64>,
    pub used_seed: Option<u64>,
}

/**
 * Run a forecast / simulation using Holt-Winters Exponential Smoothing.
 *
 * @param params Model parameters and optional historical data.
 */
pub fn simulate(params: HoltWintersParams) -> Result<HoltWintersResult, String> {
    let seed = if let Some(s) = params.seed {
        s
    } else {
        rand::rng().next_u64()
    };

    use rand::{RngCore, SeedableRng};
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    let m = params.period;
    if m == 0 {
        return Err("Period must be >= 1".to_string());
    }

    // State variables
    // L_t: Level, T_t: Trend, S_t: Seasonals (store last m values)
    let (mut l, mut t, mut s) = if let Some(ref data) = params.data {
        if data.len() < 2 * m {
            // Heuristic fallback if not enough data for proper initialization,
            // but let's try to handle at least basic cases.
            // If extremely short data, we might just default.
            if data.len() < 2 {
                return Err("Data length must be at least 2 for trend initialization".to_string());
            }
        }
        initialize_from_data(&data, m, params.seasonal_type)?
    } else {
        // Default warm-up state if no data provided
        let initial_l = 100.0;
        let initial_t = 0.0;
        let initial_s = match params.seasonal_type {
            SeasonalType::Additive => vec![0.0; m],
            SeasonalType::Multiplicative => vec![1.0; m],
        };
        (initial_l, initial_t, initial_s)
    };

    // If data is provided, we might want to "run" the filter through the data to get the final state
    // before simulating 'steps'.
    // If no data, we just start generating.

    // BUT, the standard usage in this project (like arima/garch) seems to be:
    // 1. If params.data is present, fit/warmup on it, then project 'steps' ahead.
    // 2. The output 'path' usually contains ONLY the future 'steps' points?
    //    Let's check arima.rs:
    //    "let result_path = integrated_x.into_iter().skip(n - params.steps).collect();" -> Yes, it returns the accumulated path.
    //    Wait, arima.rs `if let Some(ref data)` block (lines 93+) seems to append predictions to the data?
    //    Let's look closely at `arima.rs` Logic:
    //    It returns `current_predictions` which accumulates `next_preds`.
    //    Wait, `ArimaResult.path` is just the predictions?
    //    Line 94: `let mut current_predictions = result_x[start_idx..].to_vec();`
    //    It seems it calculates the 'diff' predictions then integrates them back.
    //    Effectively, it returns the *continuation* of the series.

    // So for ES:
    // We update state (L, T, S) iterating through `params.data` (if any).
    // Then we generate `steps` future values, updating state stochastically if sigma > 0.

    if let Some(ref data) = params.data {
        // Run filter over history to update L, T, S
        for i in 0..data.len() {
            let y = data[i];
            // let s_idx = i % m; // Unused

            // We need S_{t-m}. In our circular buffer or vec logic, we need to be careful.
            // Let's store S as a Vec of length m, representing indices [t-m, ..., t-1] relative to current t?
            // Actually, standard HW formulation:
            // S_t computed at time t is used for time t+m.
            // When at time t, we use S_{t-m} (which was computed m steps ago).
            // Let's keep `s` as a Vec<f64> of size m. `s[i % m]` will store the seasonal component for season `i`.
            // Wait, this is tricky for updates.
            // Easier: Just keep track of the *last known* seasonal index.

            // Standard approach:
            // S_new = gamma * (Y - L_new) + (1-gamma) * S_old  (Additive)
            // But we need the S from "last cycle".
            // Let's interpret `s` as: `s[k]` is the Seasonal factor for the k-th period of the cycle (0..m-1).
            // At step i, the season index is `idx = i % m`.
            // We use `s[idx]` as the seasonal component coming from the past.
            // Then we update `s[idx]` with the new value.

            let idx = i % m;
            let old_s = s[idx];
            let (new_l, new_t, new_s) = match params.seasonal_type {
                SeasonalType::Additive => {
                    let l_val = params.alpha * (y - old_s) + (1.0 - params.alpha) * (l + t);
                    let t_val = params.beta * (l_val - l) + (1.0 - params.beta) * t;
                    let s_val = params.gamma * (y - l_val) + (1.0 - params.gamma) * old_s;
                    (l_val, t_val, s_val)
                }
                SeasonalType::Multiplicative => {
                    // Avoid division by zero
                    let safe_old_s = if old_s.abs() < 1e-9 { 1.0 } else { old_s };
                    let l_val = params.alpha * (y / safe_old_s) + (1.0 - params.alpha) * (l + t);
                    let t_val = params.beta * (l_val - l) + (1.0 - params.beta) * t;
                    let s_val = params.gamma * (y / if l_val.abs() < 1e-9 { 1.0 } else { l_val })
                        + (1.0 - params.gamma) * safe_old_s;
                    (l_val, t_val, s_val)
                }
            };
            l = new_l;
            t = new_t;
            s[idx] = new_s;
        }
    } else {
        // If no data, we are already at "end" of initialization (t=0 effectively).
        // But we might want some warmup steps if sigma > 0 to let randomness diverge?
        // For now, assume parameters start from specific L, T, S.
    }

    // Forecasting / Simulation
    let mut predictions = Vec::with_capacity(params.steps);
    let start_idx = params.data.as_ref().map(|d| d.len()).unwrap_or(0);

    // We continue updating the state step by step, adding noise.
    // Or do we just project? "Simulation" usually implies adding noise to the recursive equations.
    // Standard HW forecasting projects the expected value.
    // "Simulating" paths implies: Y_{t+h} = L_{t+h-1} + T_{t+h-1} + S_{t+h-m} + \epsilon
    // Then UPADTE L, T, S using this simulated Y.

    for i in 0..params.steps {
        let current_t = start_idx + i;
        let idx = current_t % m;
        let old_s = s[idx];

        // 1. Generate Prediction / Value for this step
        let noise: f64 = StandardNormal.sample(&mut rng);
        let epsilon = noise * params.sigma;

        let y_sim = match params.seasonal_type {
            SeasonalType::Additive => l + t + old_s + epsilon,
            SeasonalType::Multiplicative => (l + t) * old_s + epsilon, // Noise usually additive? Or multiplicative?
                                                                       // Typically additive noise for the observation equation.
        };

        predictions.push(y_sim);

        // 2. Update states (Feedback Loop)
        // If we simply "forecast", we fix states. But "simulation" implies evolving the process with noise.
        // We feed y_sim back into the equations.
        let (new_l, new_t, new_s) = match params.seasonal_type {
            SeasonalType::Additive => {
                let l_val = params.alpha * (y_sim - old_s) + (1.0 - params.alpha) * (l + t);
                let t_val = params.beta * (l_val - l) + (1.0 - params.beta) * t;
                let s_val = params.gamma * (y_sim - l_val) + (1.0 - params.gamma) * old_s;
                (l_val, t_val, s_val)
            }
            SeasonalType::Multiplicative => {
                let safe_old_s = if old_s.abs() < 1e-9 { 1.0 } else { old_s };
                let l_val = params.alpha * (y_sim / safe_old_s) + (1.0 - params.alpha) * (l + t);
                let t_val = params.beta * (l_val - l) + (1.0 - params.beta) * t;
                let s_val = params.gamma * (y_sim / if l_val.abs() < 1e-9 { 1.0 } else { l_val })
                    + (1.0 - params.gamma) * safe_old_s;
                (l_val, t_val, s_val)
            }
        };

        l = new_l;
        t = new_t;
        s[idx] = new_s;
    }

    Ok(HoltWintersResult {
        path: predictions,
        used_seed: Some(seed),
    })
}

// Simple heuristic initialization
fn initialize_from_data(
    data: &[f64],
    m: usize,
    stype: SeasonalType,
) -> Result<(f64, f64, Vec<f64>), String> {
    if data.len() < m {
        // Fallback for very short data (should verify earlier)
        return match stype {
            SeasonalType::Additive => Ok((axis_mean(data), 0.0, vec![0.0; m])),
            SeasonalType::Multiplicative => Ok((axis_mean(data), 0.0, vec![1.0; m])),
        };
    }

    // Initial Level = Average of first season
    let mut initial_l = 0.0;
    for i in 0..m {
        initial_l += data[i];
    }
    initial_l /= m as f64;

    // Initial Trend
    // Simple heuristic: (Average of last season - Average of first season) / (N - m) roughly?
    // Or just (data[m] - data[0]) / m if we have at least m+1 points?
    // Let's use standard Hyndman-Khandakar or similar initialization if possible, but keep it simple.
    // Let's try: Trend = (Mean of second season - Mean of first season) / m
    // Only if we have 2*m data points.
    let initial_t = if data.len() >= 2 * m {
        let mut sum_first = 0.0;
        let mut sum_second = 0.0;
        for i in 0..m {
            sum_first += data[i];
            sum_second += data[i + m];
        }
        (sum_second - sum_first) / (m as f64 * m as f64)
    } else {
        0.0 // Insufficient data for reliable trend
    };

    // Initial Seasonals
    let mut initial_s = Vec::with_capacity(m);
    for i in 0..m {
        match stype {
            SeasonalType::Additive => {
                // S_i = y_i - L
                // Better: S_i = y_i - (L_0 + T_0 * (i+1)) ?
                // Simply difference from mean of first season for now.
                initial_s.push(data[i] - initial_l);
            }
            SeasonalType::Multiplicative => {
                if initial_l.abs() < 1e-9 {
                    initial_s.push(1.0);
                } else {
                    initial_s.push(data[i] / initial_l);
                }
            }
        }
    }

    // Normalize seasonals
    // Additive: sum(S) should be 0.
    // Multiplicative: sum(S) should be m.
    match stype {
        SeasonalType::Additive => {
            let sum_s: f64 = initial_s.iter().sum();
            let correction = sum_s / m as f64;
            for val in initial_s.iter_mut() {
                *val -= correction;
            }
        }
        SeasonalType::Multiplicative => {
            let sum_s: f64 = initial_s.iter().sum();
            if sum_s.abs() > 1e-9 {
                let correction = m as f64 / sum_s;
                for val in initial_s.iter_mut() {
                    *val *= correction;
                }
            }
        }
    }

    Ok((initial_l, initial_t, initial_s))
}

fn axis_mean(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    data.iter().sum::<f64>() / data.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hw_determinism() {
        let params = HoltWintersParams {
            alpha: 0.5,
            beta: 0.1,
            gamma: 0.1,
            period: 4,
            seasonal_type: SeasonalType::Additive,
            steps: 10,
            sigma: 1.0,
            seed: Some(42),
            data: Some(vec![10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0]),
        };

        let res1 = simulate(params.clone()).unwrap();
        let res2 = simulate(params.clone()).unwrap();

        assert_eq!(res1.path, res2.path);
    }

    #[test]
    fn test_hw_multiplicative_basic() {
        // Multiplicative seasonality often used for exponentially growing series or variable amplitude
        let params = HoltWintersParams {
            alpha: 0.5,
            beta: 0.1,
            gamma: 0.1,
            period: 4,
            seasonal_type: SeasonalType::Multiplicative,
            steps: 5,
            sigma: 0.0, // No noise to check trend
            seed: None,
            data: Some(vec![10.0, 20.0, 10.0, 20.0, 10.0, 20.0, 10.0, 20.0]),
        };

        let res = simulate(params).unwrap();
        assert_eq!(res.path.len(), 5);
        // Expect pattern roughly 10, 20, ...
        // Check first simulated value close to 10
        assert!((res.path[0] - 10.0).abs() < 5.0);
    }
}
