use crate::simulation::options::OptionsMarket;
use rand::prelude::*;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

/// Types of scenarios that can be applied to the market
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Scenario {
    /// Immediate price shock to an asset
    PriceShock { asset: String, magnitude: f64 }, // e.g., 0.10 for +10%, -0.05 for -5%
    /// Immediate volatility spike
    VolatilitySpike { asset: String, magnitude: f64 }, // e.g., 0.20 for +20% absolute vol
    /// Market crash scenario (combined shock and vol spike)
    MarketCrash { asset: String, severity: f64 },
}

/// Engine for running scenarios against an OptionsMarket
pub struct ScenarioEngine {
    pub initial_market: OptionsMarket,
}

impl ScenarioEngine {
    pub fn new(market: OptionsMarket) -> Self {
        Self {
            initial_market: market,
        }
    }

    /// Apply a scenario and return the modified market state
    pub fn apply_scenario(&self, scenario: &Scenario) -> OptionsMarket {
        let mut market = self.initial_market.clone(); // In real usage, cloning might be expensive

        match scenario {
            Scenario::PriceShock { asset, magnitude } => {
                if let Some(price) = market.underlying_price.get_mut(asset) {
                    *price *= 1.0 + magnitude;
                }
            }
            Scenario::VolatilitySpike { asset, magnitude } => {
                if let Some(vol) = market.underlying_volatility.get_mut(asset) {
                    *vol += magnitude;
                }
            }
            Scenario::MarketCrash { asset, severity } => {
                // Severity 1.0 = 10% drop, 20% vol spike
                if let Some(price) = market.underlying_price.get_mut(asset) {
                    *price *= 1.0 - (0.10 * severity);
                }
                if let Some(vol) = market.underlying_volatility.get_mut(asset) {
                    *vol += 0.20 * severity;
                }
            }
        }

        // Must update Greeks after changing underlying parameters
        // We assume timestamp 0 for simplicity or pass it in
        market.update_greeks(0);

        market
    }
}

/// Monte Carlo simulation results
#[derive(Debug)]
pub struct MonteCarloResult {
    pub final_price_mean: f64,
    pub final_price_std: f64,
    pub var_95: f64,          // Value at Risk (95%)
    pub cvar_95: f64,         // Conditional VaR (95%)
    pub paths: Vec<Vec<f64>>, // Optional: store full paths
}

/// Monte Carlo path generator
pub struct MonteCarlo {
    pub num_simulations: usize,
    pub time_steps: usize,
    pub dt: f64, // Time step in years
}

impl MonteCarlo {
    pub fn new(num_simulations: usize, time_steps: usize, dt: f64) -> Self {
        Self {
            num_simulations,
            time_steps,
            dt,
        }
    }

    /// Run GBM simulation for a single asset
    pub fn run(&self, initial_price: f64, drift: f64, vol: f64) -> MonteCarloResult {
        let mut rng = rand::rng();
        let normal = Normal::new(0.0, 1.0).unwrap();

        let mut final_prices = Vec::with_capacity(self.num_simulations);
        let mut all_paths = Vec::new(); // Only store if needed, can be heavy

        for _ in 0..self.num_simulations {
            let mut price = initial_price;
            let mut path = Vec::with_capacity(self.time_steps + 1);
            path.push(price);

            for _ in 0..self.time_steps {
                let z = normal.sample(&mut rng);
                // Geometric Brownian Motion: dS = S * (mu * dt + sigma * dW)
                // Discretized: S(t+dt) = S(t) * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
                let change = (drift - 0.5 * vol * vol) * self.dt + vol * self.dt.sqrt() * z;
                price *= change.exp();
                path.push(price);
            }
            final_prices.push(price);
            all_paths.push(path);
        }

        // Calculate statistics
        let mean: f64 = final_prices.iter().sum::<f64>() / self.num_simulations as f64;
        let variance: f64 = final_prices
            .iter()
            .map(|&p| (p - mean).powi(2))
            .sum::<f64>()
            / self.num_simulations as f64;

        // VaR / CVaR
        // Sort final prices
        final_prices.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let index_5pct = (self.num_simulations as f64 * 0.05) as usize;
        let var_price = final_prices[index_5pct];
        let var_95 = initial_price - var_price;

        let tail_sum: f64 = final_prices.iter().take(index_5pct).sum();
        let cvar_price = if index_5pct > 0 {
            tail_sum / index_5pct as f64
        } else {
            var_price
        };
        let cvar_95 = initial_price - cvar_price;

        MonteCarloResult {
            final_price_mean: mean,
            final_price_std: variance.sqrt(),
            var_95,
            cvar_95,
            paths: all_paths,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulation::options::{OptionContract, OptionType};

    #[test]
    fn test_scenario_application() {
        let mut market = OptionsMarket::new(0.05);
        market.set_underlying_price("BTC", 50000.0);
        market.set_underlying_volatility("BTC", 0.5);

        let contract = OptionContract {
            id: "BTC-50k-C".to_string(),
            underlying_symbol: "BTC".to_string(),
            strike: 50000.0,
            expiry: 1000,
            contract_type: OptionType::Call,
        };
        market.add_contract(contract);
        market.update_greeks(0);

        let engine = ScenarioEngine::new(market);

        // 1. Price Shock +10%
        let scenario = Scenario::PriceShock {
            asset: "BTC".to_string(),
            magnitude: 0.10,
        };
        let shocked_market = engine.apply_scenario(&scenario);

        let price = *shocked_market.underlying_price.get("BTC").unwrap();
        assert!(
            (price - 55000.0).abs() < 1e-6,
            "Price {} not equal to 55000.0",
            price
        );

        // Greeks should update - Call Delta should increase as we are now deeper ITM
        // Note: Initial Delta (ATM) ~ 0.5. At 55k (ITM), Delta > 0.5
        let initial_greeks = engine.initial_market.greeks.get("BTC-50k-C").unwrap();
        let shocked_greeks = shocked_market.greeks.get("BTC-50k-C").unwrap();

        assert!(shocked_greeks.delta > initial_greeks.delta);
    }

    #[test]
    fn test_monte_carlo() {
        let mc = MonteCarlo::new(1000, 252, 1.0 / 252.0); // 1 year, daily steps
        let res = mc.run(100.0, 0.05, 0.2);

        // Mean should be close to initial * exp(mu * T) = 100 * exp(0.05) ~= 105.12
        assert!(res.final_price_mean > 100.0 && res.final_price_mean < 110.0);

        // VaR should be positive (potential loss)
        assert!(res.var_95 > 0.0);
        assert!(res.cvar_95 >= res.var_95);
    }
}
