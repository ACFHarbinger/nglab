/*!
 * Generalized AutoRegressive Conditional Heteroskedasticity (GARCH) modeling.
 *
 * Implements GARCH(p,q) for volatility clustering
 * and conditional variance estimation in financial returns.
 */

use rand_distr::{Distribution, StandardNormal};
use serde::{Deserialize, Serialize};

use crate::errors::{ArenaError, ArenaResult};

/**
 * Configuration parameters for GARCH simulation and estimation.
 */
#[derive(Serialize, Deserialize, Debug)]
pub struct GarchParams {
    /// Constant variance term (omega).
    pub omega: f64,
    /// ARCH coefficients (alpha) for lagged squared returns.
    pub alpha: Vec<f64>,
    /// GARCH coefficients (beta) for lagged variances.
    pub beta: Vec<f64>,
    /// Number of steps to simulate.
    pub steps: usize,
    /// Optional seed for reproducibility.
    pub seed: Option<u64>,
    /// Optional historical returns for process initialization.
    pub data: Option<Vec<f64>>,
}

/**
 * Results of the GARCH simulation, including returns and conditional volatility.
 */
#[derive(Serialize, Deserialize, Debug)]
pub struct GarchResult {
    /// The simulated returns series.
    pub returns: Vec<f64>,
    /// The conditional volatility (sigma) for each step.
    pub volatility: Vec<f64>,
}

/// Simulates a GARCH(p,q) process for the specified number of steps.
///
/// # Arguments
/// * `params` - GARCH model configuration and simulation parameters.
pub fn simulate(params: GarchParams) -> ArenaResult<GarchResult> {
    let mut rng = if let Some(s) = params.seed {
        use rand::SeedableRng;
        rand::rngs::StdRng::seed_from_u64(s)
    } else {
        use rand::SeedableRng;
        rand::rngs::StdRng::from_rng(&mut rand::rng())
    };

    let p = params.alpha.len();
    let q = params.beta.len();
    let max_lag = p.max(q);

    let (mut epsilon, mut sigma2, start_idx) = if let Some(ref data) = params.data {
        if data.len() < max_lag {
            return Err(ArenaError::ModelError(format!(
                "Data length must be at least {}",
                max_lag
            )));
        }
        let n = params.steps + data.len();
        let mut eps = vec![0.0; n];
        let mut sig2 = vec![0.0; n];

        // Fill initial epsilon from data
        eps[..data.len()].copy_from_slice(data);

        let sum_alpha: f64 = params.alpha.iter().sum();
        let sum_beta: f64 = params.beta.iter().sum();
        let unconditional_var = if sum_alpha + sum_beta < 1.0 {
            params.omega / (1.0 - sum_alpha - sum_beta)
        } else {
            params.omega
        };

        sig2[..data.len()].fill(unconditional_var);
        for t in max_lag..data.len() {
            let mut var = params.omega;
            for i in 0..p {
                var += params.alpha[i] * eps[t - 1 - i].powi(2);
            }
            for j in 0..q {
                var += params.beta[j] * sig2[t - 1 - j];
            }
            sig2[t] = var;
        }

        (eps, sig2, data.len())
    } else {
        let n = params.steps + max_lag + 100;
        let mut eps = vec![0.0; n];
        let mut sig2 = vec![0.0; n];
        let sum_alpha: f64 = params.alpha.iter().sum();
        let sum_beta: f64 = params.beta.iter().sum();
        let unconditional_var = if sum_alpha + sum_beta < 1.0 {
            params.omega / (1.0 - sum_alpha - sum_beta)
        } else {
            params.omega
        };

        for t in 0..max_lag {
            sig2[t] = unconditional_var;
            let z: f64 = StandardNormal.sample(&mut rng);
            eps[t] = sig2[t].sqrt() * z;
        }
        (eps, sig2, max_lag)
    };

    let n = epsilon.len();

    for t in start_idx..n {
        let mut var = params.omega;

        for i in 0..p {
            var += params.alpha[i] * epsilon[t - 1 - i].powi(2);
        }

        for j in 0..q {
            var += params.beta[j] * sigma2[t - 1 - j];
        }

        sigma2[t] = var;
        let z: f64 = StandardNormal.sample(&mut rng);
        epsilon[t] = var.sqrt() * z;
    }

    let returns = epsilon.into_iter().skip(start_idx).collect();
    let volatility = sigma2
        .into_iter()
        .skip(start_idx)
        .map(|v| v.sqrt())
        .collect();

    Ok(GarchResult {
        returns,
        volatility,
    })
}
