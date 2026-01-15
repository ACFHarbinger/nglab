//! Rough Heston stochastic volatility model.
//!
//! Implements the rough Heston model for price paths
//! with fractional Brownian motion in the volatility process.

use ndarray::Array1;
use rand_distr::{Distribution, StandardNormal};
use serde::{Deserialize, Serialize};

use crate::models::rough_bergomi::generate_fbm_cholesky;

#[derive(Serialize, Deserialize, Debug)]
pub struct RoughHestonParams {
    pub spot: f64,
    pub strike: f64,
    pub rate: f64,
    pub maturity: f64,
    pub v0: f64,
    pub theta: f64,
    pub kappa: f64,
    pub nu: f64,
    pub rho: f64,
    pub hurst: f64,
    pub steps: usize,
    pub paths: usize,
    pub option_type: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RoughHestonResult {
    pub price: f64,
    pub std_error: f64,
    pub mean_terminal: f64,
    pub p05: f64,
    pub p95: f64,
    pub paths: usize,
    pub steps: usize,
}

fn clamp_f64(value: f64, min: f64, max: f64) -> f64 {
    if value < min {
        min
    } else if value > max {
        max
    } else {
        value
    }
}

fn percentile(values: &[f64], p: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = (sorted.len() - 1) as f64 * p;
    let lower = idx.floor() as usize;
    let upper = idx.ceil() as usize;
    if lower == upper {
        sorted[lower]
    } else {
        let weight = idx - lower as f64;
        sorted[lower] * (1.0 - weight) + sorted[upper] * weight
    }
}

/**
 * Simulate price paths under the rough Heston model.
 */
pub fn simulate(params: RoughHestonParams) -> Result<RoughHestonResult, String> {
    let steps = params.steps.max(16).min(512);
    let paths = params.paths.max(100).min(10_000);
    let hurst = clamp_f64(params.hurst, 0.01, 0.49);
    let rho = clamp_f64(params.rho, -0.99, 0.99);

    if params.maturity <= 0.0 {
        return Err("Maturity must be positive.".to_string());
    }

    let dt = params.maturity / steps as f64;
    let dt_h = dt.powf(hurst);
    if dt_h == 0.0 {
        return Err("Invalid time step for rough kernel.".to_string());
    }

    let chol = generate_fbm_cholesky(steps + 1, hurst, dt)?;
    let mut rng = rand::rng();

    let mut terminal_prices = Vec::with_capacity(paths);
    let mut discounted_payoffs = Vec::with_capacity(paths);

    for _ in 0..paths {
        let z_vec: Vec<f64> = (0..=steps)
            .map(|_| StandardNormal.sample(&mut rng))
            .collect();
        let z_array = Array1::from(z_vec);
        let fbm_path = chol.dot(&z_array);

        let mut s = params.spot;
        let mut v = params.v0.max(1e-8);

        for i in 0..steps {
            let d_b = fbm_path[i + 1] - fbm_path[i];
            let variance_shock = d_b;
            v += params.kappa * (params.theta - v) * dt + params.nu * v.sqrt() * variance_shock;
            v = v.max(1e-8);

            let z: f64 = StandardNormal.sample(&mut rng);
            let d_b_std = d_b / dt_h;
            let z_corr = rho * d_b_std + (1.0 - rho * rho).sqrt() * z;
            let d_w = z_corr * dt.sqrt();

            s *= ((params.rate - 0.5 * v) * dt + v.sqrt() * d_w).exp();
        }

        terminal_prices.push(s);
        let payoff = match params.option_type.as_str() {
            "call" => (s - params.strike).max(0.0),
            "put" => (params.strike - s).max(0.0),
            other => return Err(format!("Invalid option_type: {}", other)),
        };
        discounted_payoffs.push(payoff * (-params.rate * params.maturity).exp());
    }

    let mean = discounted_payoffs.iter().sum::<f64>() / paths as f64;
    let variance = discounted_payoffs
        .iter()
        .map(|p| (p - mean).powi(2))
        .sum::<f64>()
        / ((paths - 1).max(1) as f64);
    let std_error = (variance / paths as f64).sqrt();
    let mean_terminal = terminal_prices.iter().sum::<f64>() / paths as f64;

    Ok(RoughHestonResult {
        price: mean,
        std_error,
        mean_terminal,
        p05: percentile(&terminal_prices, 0.05),
        p95: percentile(&terminal_prices, 0.95),
        paths,
        steps,
    })
}
