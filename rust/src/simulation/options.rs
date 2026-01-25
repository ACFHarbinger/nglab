use crate::models::black_scholes::{self, BlackScholesParams};
use crate::simulation::orderbook::OrderBook;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OptionType {
    Call,
    Put,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OptionContract {
    pub id: String,
    pub underlying_symbol: String,
    pub strike: f64,
    pub expiry: u64,
    pub contract_type: OptionType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Greeks {
    pub delta: f64,
    pub gamma: f64,
    pub theta: f64, // Not directly from BS result, but usually computed via finite difference or formula
    pub vega: f64,
    pub rho: f64,
}

impl Default for Greeks {
    fn default() -> Self {
        Self {
            delta: 0.0,
            gamma: 0.0,
            theta: 0.0,
            vega: 0.0,
            rho: 0.0,
        }
    }
}

#[derive(Clone)]
pub struct OptionsMarket {
    /// Map from contract ID to its OrderBook
    pub books: HashMap<String, OrderBook>,
    /// Registry of contract details
    pub contracts: HashMap<String, OptionContract>,
    /// Cached Greeks for each contract
    pub greeks: HashMap<String, Greeks>,
    /// Risk-free rate for pricing
    pub risk_free_rate: f64,
    /// Underlying volatility (simplified: one vol per underlying for now)
    pub underlying_volatility: HashMap<String, f64>,
    /// Underlying spot price
    pub underlying_price: HashMap<String, f64>,
}

impl OptionsMarket {
    pub fn new(risk_free_rate: f64) -> Self {
        Self {
            books: HashMap::new(),
            contracts: HashMap::new(),
            greeks: HashMap::new(),
            risk_free_rate,
            underlying_volatility: HashMap::new(),
            underlying_price: HashMap::new(),
        }
    }

    pub fn add_contract(&mut self, contract: OptionContract) {
        let id = contract.id.clone();
        self.books.insert(id.clone(), OrderBook::new());
        self.greeks.insert(id.clone(), Greeks::default());
        self.contracts.insert(id, contract);
    }

    pub fn set_underlying_price(&mut self, symbol: &str, price: f64) {
        self.underlying_price.insert(symbol.to_string(), price);
    }

    pub fn set_underlying_volatility(&mut self, symbol: &str, vol: f64) {
        self.underlying_volatility.insert(symbol.to_string(), vol);
    }

    /// Update Greeks for all contracts based on current spot and vol
    pub fn update_greeks(&mut self, current_time: u64) {
        for (id, contract) in &self.contracts {
            let spot = *self
                .underlying_price
                .get(&contract.underlying_symbol)
                .unwrap_or(&100.0);
            let vol = *self
                .underlying_volatility
                .get(&contract.underlying_symbol)
                .unwrap_or(&0.2);

            // Time to expiry in years
            let tte_seconds = contract.expiry.saturating_sub(current_time);
            let tte_years = tte_seconds as f64 / (365.0 * 24.0 * 3600.0);

            if tte_years <= 0.0 {
                continue;
            }

            let params = BlackScholesParams {
                spot,
                strike: contract.strike,
                rate: self.risk_free_rate,
                volatility: vol,
                maturity: tte_years,
            };

            let bs_result = black_scholes::price(params);

            // Using BS result to populate what we have
            // Note: Theta and Rho are not in the current BlackScholesResult from models::black_scholes
            // We will just use what is available for now.
            let g = Greeks {
                delta: bs_result.delta,
                gamma: bs_result.gamma,
                vega: bs_result.vega,
                theta: 0.0, // Placeholder
                rho: 0.0,   // Placeholder
            };

            self.greeks.insert(id.clone(), g);
        }
    }

    pub fn get_book_mut(&mut self, contract_id: &str) -> Option<&mut OrderBook> {
        self.books.get_mut(contract_id)
    }

    /// Exercise logic: Return list of contracts and their intrinsic value if ITM at expiry
    pub fn check_exercises(&self, current_time: u64) -> Vec<(String, f64)> {
        let mut exercises = Vec::new();
        for (id, contract) in &self.contracts {
            if contract.expiry <= current_time {
                let spot = *self
                    .underlying_price
                    .get(&contract.underlying_symbol)
                    .unwrap_or(&0.0);
                let value = match contract.contract_type {
                    OptionType::Call => (spot - contract.strike).max(0.0),
                    OptionType::Put => (contract.strike - spot).max(0.0),
                };
                if value > 0.0 {
                    exercises.push((id.clone(), value));
                }
            }
        }
        exercises
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_options_market_basics() {
        let mut market = OptionsMarket::new(0.05);
        market.set_underlying_price("BTC", 50000.0);
        market.set_underlying_volatility("BTC", 0.5);

        let contract = OptionContract {
            id: "BTC-50000-C".to_string(),
            underlying_symbol: "BTC".to_string(),
            strike: 50000.0,
            expiry: 100000,
            contract_type: OptionType::Call,
        };
        market.add_contract(contract);

        assert!(market.get_book_mut("BTC-50000-C").is_some());

        // Update Greeks with some time remaining
        market.update_greeks(0);
        let greeks = market.greeks.get("BTC-50000-C").unwrap();
        // At the money call delta should be around 0.5
        assert!(greeks.delta > 0.4 && greeks.delta < 0.7);
    }

    #[test]
    fn test_exercise() {
        let mut market = OptionsMarket::new(0.05);
        market.set_underlying_price("ETH", 3000.0);

        let c1 = OptionContract {
            id: "ETH-2000-C".to_string(), // Deep ITM
            underlying_symbol: "ETH".to_string(),
            strike: 2000.0,
            expiry: 100,
            contract_type: OptionType::Call,
        };
        market.add_contract(c1);

        let c2 = OptionContract {
            id: "ETH-4000-C".to_string(), // OTM
            underlying_symbol: "ETH".to_string(),
            strike: 4000.0,
            expiry: 100,
            contract_type: OptionType::Call,
        };
        market.add_contract(c2);

        // Not expired yet
        let exercises = market.check_exercises(50);
        assert!(exercises.is_empty());

        // Expired
        let exercises = market.check_exercises(100);
        assert_eq!(exercises.len(), 1);
        assert_eq!(exercises[0].0, "ETH-2000-C");
        assert_eq!(exercises[0].1, 1000.0); // 3000 - 2000
    }
}
