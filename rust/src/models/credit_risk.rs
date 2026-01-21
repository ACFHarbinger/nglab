/*!
 * Credit risk modeling and CVA calculation.
 *
 * Implements basic Merton-style structural credit risk models
 * to adjust derivative prices for counterparty risk.
 */

use serde::{Deserialize, Serialize};

use crate::models::black_scholes::{price as bsm_price, BlackScholesParams};

/**
 * Parameters for the structural credit risk model.
 */
#[derive(Serialize, Deserialize, Debug)]
pub struct CreditRiskParams {
    /// Underlying asset spot price.
    pub spot: f64,
    /// Option strike price.
    pub strike: f64,
    /// Risk-free rate.
    pub rate: f64,
    /// Annualized volatility.
    pub volatility: f64,
    /// Time to maturity.
    pub maturity: f64,
    /// Type of option ("call" or "put").
    pub option_type: String,
    /// Hazard rate for default probability.
    pub hazard_rate: f64,
    /// Recovery rate in case of default.
    pub recovery: f64,
}

/**
 * Results of the credit risk adjustment calculation.
 */
#[derive(Serialize, Deserialize, Debug)]
pub struct CreditRiskResult {
    /// Base option price without credit risk.
    pub base: f64,
    /// Credit-adjusted option price.
    pub adjusted: f64,
    /// Probability of survival until maturity.
    pub survival: f64,
    /// Credit Valuation Adjustment (CVA).
    pub cva: f64,
}

/// Calculates the credit-adjusted price and Credit Valuation Adjustment (CVA).
///
/// This uses a Merton-style structural model with a constant hazard rate.
pub fn price(params: CreditRiskParams) -> Result<CreditRiskResult, String> {
    let bsm = bsm_price(BlackScholesParams {
        spot: params.spot,
        strike: params.strike,
        rate: params.rate,
        volatility: params.volatility,
        maturity: params.maturity,
    });

    let base = match params.option_type.as_str() {
        "call" => bsm.call,
        "put" => bsm.put,
        other => return Err(format!("Invalid option_type: {}", other)),
    };

    let survival = (-params.hazard_rate * params.maturity.max(1e-8)).exp();
    let adjusted = base * (survival + params.recovery * (1.0 - survival));
    let cva = base - adjusted;

    Ok(CreditRiskResult {
        base,
        adjusted,
        survival,
        cva,
    })
}
