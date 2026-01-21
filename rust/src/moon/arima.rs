/*!
 * AutoRegressive Integrated Moving Average (ARIMA) simulation.
 *
 * Provides a simplified ARIMA(p,d,q) model for generating
 * synthetic time series data.
 */

use rand_distr::{Distribution, StandardNormal};
use serde::{Deserialize, Serialize};
use ts_rs::TS;

/**
 * Parameters for the ARIMA simulation.
 */
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct ArimaParams {
    pub ar: Vec<f64>, // AR coefficients (phi)
    pub ma: Vec<f64>, // MA coefficients (theta)
    pub d: usize,     // Integration order
    pub steps: usize,
    pub sigma: f64, // Noise standard deviation
    pub seed: Option<u64>,
    pub data: Option<Vec<f64>>, // Past data
}

/**
 * Result containing the simulated ARIMA path.
 */
#[derive(Serialize, Deserialize, Debug, TS)]
#[ts(export)]
pub struct ArimaResult {
    pub path: Vec<f64>,
    pub used_seed: Option<u64>,
}

use crate::errors::{ArenaError, ArenaResult};
use crate::utils::math::safe_div;

/**
 * Fit an ARIMA(p,d,q) model to data and simulate future steps.
 *
 * Uses Yule-Walker equations for AR estimation.
 */
pub fn fit_and_simulate(
    data: Vec<f64>,
    p: usize,
    d: usize,
    q: usize,
    steps: usize,
) -> ArenaResult<ArimaResult> {
    if data.len() < p + d + 2 {
        return Err(ArenaError::ModelError(format!(
            "Insufficient data for ARIMA({},{},{}). Need at least {} points.",
            p,
            d,
            q,
            p + d + 2
        )));
    }

    // 1. Differencing (Integrated part)
    let mut current = data.clone();
    for _ in 0..d {
        let mut diff = Vec::new();
        for i in 1..current.len() {
            diff.push(current[i] - current[i - 1]);
        }
        current = diff;
    }

    // 2. Estimate AR coefficients (phi) using Yule-Walker
    let ar_coeffs = if p > 0 {
        estimate_ar_yule_walker(&current, p)?
    } else {
        Vec::new()
    };

    // 3. Estimate MA coefficients (theta)
    // Simplified: we use zeros for MA as full estimation (MLE/CSS) is significantly more complex.
    let ma_coeffs = vec![0.0; q];

    // 4. Estimate sigma (residual variance)
    let sigma = 0.01; // Default or estimate from residuals

    let params = ArimaParams {
        ar: ar_coeffs,
        ma: ma_coeffs,
        d,
        steps,
        sigma,
        seed: None,
        data: Some(data),
    };

    simulate(params)
}

fn estimate_ar_yule_walker(data: &[f64], p: usize) -> ArenaResult<Vec<f64>> {
    let n = data.len();
    let mean = data.iter().sum::<f64>() / n as f64;
    let centered: Vec<f64> = data.iter().map(|&x| x - mean).collect();

    // Autocovariances gamma(k)
    let mut gamma = vec![0.0; p + 1];
    for k in 0..=p {
        let mut sum = 0.0;
        for i in k..n {
            sum += centered[i] * centered[i - k];
        }
        gamma[k] = sum / n as f64;
    }

    if gamma[0].abs() < 1e-12 {
        return Ok(vec![0.0; p]);
    }

    // Solve Yule-Walker equations: R * phi = g
    // where R is Toeplitz matrix of autocovariances
    let mut r_mat = ndarray::Array2::zeros((p, p));
    let mut g_vec = ndarray::Array1::zeros(p);

    for i in 0..p {
        g_vec[i] = gamma[i + 1];
        for j in 0..p {
            let lag = (i as i32 - j as i32).unsigned_abs() as usize;
            r_mat[[i, j]] = gamma[lag];
        }
    }

    // Solve using ndarray-linalg or simple Cramer/Levinson-Durbin
    // Since we don't want extra dependencies, let's use a simple Gaussian elimination for small p.
    solve_linear_system(r_mat, g_vec)
}

fn solve_linear_system(
    mut a: ndarray::Array2<f64>,
    mut b: ndarray::Array1<f64>,
) -> ArenaResult<Vec<f64>> {
    let n = b.len();
    for i in 0..n {
        // Pivot
        let mut max_row = i;
        for k in i + 1..n {
            if a[[k, i]].abs() > a[[max_row, i]].abs() {
                max_row = k;
            }
        }

        // Swap rows in A and B
        for k in i..n {
            let tmp = a[[i, k]];
            a[[i, k]] = a[[max_row, k]];
            a[[max_row, k]] = tmp;
        }
        let tmp = b[i];
        b[i] = b[max_row];
        b[max_row] = tmp;

        if a[[i, i]].abs() < 1e-12 {
            return Err(ArenaError::NumericalError(
                "Singular matrix in Yule-Walker".to_string(),
            ));
        }

        // Eliminate
        for k in i + 1..n {
            let f = safe_div(a[[k, i]], a[[i, i]], 0.0);
            for j in i..n {
                a[[k, j]] -= f * a[[i, j]];
            }
            b[k] -= f * b[i];
        }
    }

    // Back substitution
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = 0.0;
        for j in i + 1..n {
            sum += a[[i, j]] * x[j];
        }
        x[i] = safe_div(b[i] - sum, a[[i, i]], 0.0);
    }
    Ok(x)
}

/**
 * Simulate an ARIMA(p,d,q) process for the specified number of steps.
 *
 * @param params ARIMA model and simulation parameters.
 */
pub fn simulate(params: ArimaParams) -> ArenaResult<ArimaResult> {
    let seed = if let Some(s) = params.seed {
        s
    } else {
        rand::rng().next_u64()
    };

    use rand::{RngCore, SeedableRng};
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    let p = params.ar.len();
    let q = params.ma.len();

    // If data is provided, we use it to initialize.
    // Otherwise, we start from zero with a warmup.
    let initial_series = if let Some(ref data) = params.data {
        if data.len() <= params.d {
            return Err(ArenaError::ModelError(
                "Data length must be greater than integration order d".to_string(),
            ));
        }
        let mut current = data.clone();
        for _ in 0..params.d {
            let mut diff = Vec::new();
            for i in 1..current.len() {
                diff.push(current[i] - current[i - 1]);
            }
            current = diff;
        }
        current
    } else {
        vec![0.0; p.max(q).max(params.d) + 100] // Warmup
    };

    let n = params.steps + initial_series.len();

    // Generate white noise (epsilon)
    let mut eps = vec![0.0; n];
    for item in eps.iter_mut().take(n) {
        let z: f64 = StandardNormal.sample(&mut rng);
        *item = z * params.sigma;
    }

    // Generate ARMA process
    let mut x = vec![0.0; n];
    // Copy initial series
    x[..initial_series.len()].copy_from_slice(&initial_series[..]);

    let start_idx = initial_series.len();

    for t in start_idx..n {
        let mut val = eps[t];

        // AR part
        for i in 0..p {
            if t > i {
                val += params.ar[i] * x[t - 1 - i];
            }
        }

        // MA part
        for j in 0..q {
            if t > j {
                val += params.ma[j] * eps[t - 1 - j];
            }
        }

        x[t] = val;
    }

    // Integrate d times
    let result_x = x;
    if let Some(ref data) = params.data {
        let mut current_predictions = result_x[start_idx..].to_vec();

        for d_iter in 0..params.d {
            // Get the series at one level less of differencing
            let mut level_data = data.clone();
            for _ in 0..(params.d - 1 - d_iter) {
                let mut diff = Vec::new();
                for i in 1..level_data.len() {
                    diff.push(level_data[i] - level_data[i - 1]);
                }
                level_data = diff;
            }

            let mut last_val = *level_data.last().ok_or_else(|| {
                ArenaError::InternalError(
                    "Level data unexpectedly empty during integration".to_string(),
                )
            })?;
            let mut next_preds = Vec::new();
            for p_val in current_predictions {
                last_val += p_val;
                next_preds.push(last_val);
            }
            current_predictions = next_preds;
        }
        Ok(ArimaResult {
            path: current_predictions,
            used_seed: Some(seed),
        })
    } else {
        let mut integrated_x = result_x;
        for _ in 0..params.d {
            let mut integrated = vec![0.0; n];
            let mut cumsum = 0.0;
            for t in 0..n {
                cumsum += integrated_x[t];
                integrated[t] = cumsum;
            }
            integrated_x = integrated;
        }
        let result_path = integrated_x.into_iter().skip(n - params.steps).collect();
        Ok(ArimaResult {
            path: result_path,
            used_seed: Some(seed),
        })
    }
}

#[cfg(test)]
mod tests {
    //! AutoRegressive Integrated Moving Average (ARIMA) simulation.
    //!
    //! Provides a simplified ARIMA(p,d,q) model for generating
    //! synthetic time series data.

    use super::*;

    #[test]
    fn test_arima_determinism() {
        let params = ArimaParams {
            ar: vec![0.5],
            ma: vec![0.5],
            d: 1,
            steps: 10,
            sigma: 1.0,
            seed: Some(42),
            data: None,
        };

        let result1 = simulate(ArimaParams {
            seed: Some(42),
            ..params.clone()
        })
        .unwrap();
        let result2 = simulate(ArimaParams {
            seed: Some(42),
            ..params
        })
        .unwrap();

        assert_eq!(
            result1.path, result2.path,
            "ARIMA simulation should be deterministic with fixed seed"
        );
        assert_eq!(result1.used_seed, Some(42));
    }

    #[test]
    fn test_fit_and_simulate() {
        use rand::SeedableRng;
        use rand_distr::Distribution;

        // Generate simple random walk data
        let mut data = vec![100.0];
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        for _ in 0..100 {
            let noise: f64 = match StandardNormal.sample(&mut rng) {
                x => x,
            };
            let last = *data.last().unwrap();
            data.push(last + noise);
        }

        // Test fitting ARIMA(1,1,0)
        let result = fit_and_simulate(data.clone(), 1, 1, 0, 10);

        assert!(result.is_ok(), "Fitting should succeed");
        let sim = result.unwrap();
        assert_eq!(sim.path.len(), 10, "Should generate 10 steps");

        // Ensure values are not NaN or Infinite
        for val in sim.path {
            assert!(val.is_finite(), "Simulation values should be finite");
        }
    }
}
