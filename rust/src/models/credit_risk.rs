use serde::{Deserialize, Serialize};

use crate::models::black_scholes::{price as bsm_price, BlackScholesParams};

#[derive(Serialize, Deserialize, Debug)]
pub struct CreditRiskParams {
    pub spot: f64,
    pub strike: f64,
    pub rate: f64,
    pub volatility: f64,
    pub maturity: f64,
    pub option_type: String,
    pub hazard_rate: f64,
    pub recovery: f64,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CreditRiskResult {
    pub base: f64,
    pub adjusted: f64,
    pub survival: f64,
    pub cva: f64,
}

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
