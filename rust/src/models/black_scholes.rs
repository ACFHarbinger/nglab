//! Black-Scholes-Merton option pricing.
//!
//! Provides closed-form solutions for European vanilla options
//! and their Greeks (Delta, Gamma, Vega).

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
pub struct BlackScholesParams {
    pub spot: f64,
    pub strike: f64,
    pub rate: f64,
    pub volatility: f64,
    pub maturity: f64,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct BlackScholesResult {
    pub call: f64,
    pub put: f64,
    pub delta: f64,
    pub gamma: f64,
    pub vega: f64,
    pub d1: f64,
    pub d2: f64,
}

fn erf(x: f64) -> f64 {
    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let abs_x = x.abs();
    let a1 = 0.254_829_592;
    let a2 = -0.284_496_736;
    let a3 = 1.421_413_741;
    let a4 = -1.453_152_027;
    let a5 = 1.061_405_429;
    let p = 0.327_591_1;

    let t = 1.0 / (1.0 + p * abs_x);
    let y = 1.0 - (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t) * (-abs_x * abs_x).exp();
    sign * y
}

fn norm_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / 2.0_f64.sqrt()))
}

/**
 * Price European vanilla options and calculate Greeks.
 */
pub fn price(params: BlackScholesParams) -> BlackScholesResult {
    let safe_vol = params.volatility.max(1e-8);
    let safe_mat = params.maturity.max(1e-8);
    let sqrt_t = safe_mat.sqrt();
    let d1 =
        (params.spot / params.strike).ln() + (params.rate + 0.5 * safe_vol * safe_vol) * safe_mat;
    let d1 = d1 / (safe_vol * sqrt_t);
    let d2 = d1 - safe_vol * sqrt_t;

    let call =
        params.spot * norm_cdf(d1) - params.strike * (-params.rate * safe_mat).exp() * norm_cdf(d2);
    let put = params.strike * (-params.rate * safe_mat).exp() * norm_cdf(-d2)
        - params.spot * norm_cdf(-d1);
    let delta = norm_cdf(d1);
    let gamma = (-0.5 * d1 * d1).exp()
        / (params.spot * safe_vol * sqrt_t * (2.0 * std::f64::consts::PI).sqrt());
    let vega = params.spot * (-0.5 * d1 * d1).exp() * sqrt_t / (2.0 * std::f64::consts::PI).sqrt();

    BlackScholesResult {
        call,
        put,
        delta,
        gamma,
        vega,
        d1,
        d2,
    }
}
