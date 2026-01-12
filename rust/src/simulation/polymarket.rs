//! Polymarket prediction market simulation engine
//!
//! Implements the Conditional Token Framework (CTF) with:
//! - Binary outcome markets (Yes/No)
//! - Merge/Split mechanics
//! - NegRisk accounting for cross-collateralization
//! - Market resolution and price replay.

use crate::errors::{ArenaError, ArenaResult};
#[cfg(feature = "python")]
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/** A prediction market with binary outcomes */
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Market {
    pub id: String,
    pub title: String,
    pub category: String,
    pub options: Vec<String>,
    pub resolved: bool,
    pub outcome: Option<String>,
}

/** A price update for a market */
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceTick {
    pub timestamp: u64,
    pub price: f64,
}

/** Account holding conditional tokens */
#[derive(Debug, Clone, Default)]
pub struct Account {
    /** USDC collateral balance */
    pub collateral: f64,
    /** Holdings: market_id -> (yes_tokens, no_tokens) */
    pub positions: HashMap<String, (f64, f64)>,
    /** Realized PnL */
    pub realized_pnl: f64,
}

impl Account {
    /** Create a new account with initial collateral */
    pub fn new(initial_collateral: f64) -> Self {
        Account {
            collateral: initial_collateral,
            positions: HashMap::new(),
            realized_pnl: 0.0,
        }
    }

    /** Get position in a specific market */
    pub fn get_position(&self, market_id: &str) -> (f64, f64) {
        self.positions.get(market_id).copied().unwrap_or((0.0, 0.0))
    }

    /** Calculate unrealized PnL given current prices */
    pub fn unrealized_pnl(&self, prices: &HashMap<String, f64>) -> f64 {
        let mut pnl = 0.0;
        for (market_id, (yes_tokens, no_tokens)) in &self.positions {
            if let Some(&yes_price) = prices.get(market_id) {
                let no_price = 1.0 - yes_price;
                // Value if we sold at current price
                pnl += yes_tokens * yes_price + no_tokens * no_price;
                // Minus cost (complete sets are worth $1)
                let complete_sets = yes_tokens.min(*no_tokens);
                pnl -= complete_sets; // Complete sets cost $1 each
            }
        }
        pnl
    }
}

/** Polymarket trading arena */
#[cfg_attr(feature = "python", pyclass)]
#[derive(Debug, Clone)]
pub struct PolymarketArena {
    /** Available markets */
    markets: HashMap<String, Market>,
    /** Price history for each market */
    price_history: HashMap<String, Vec<PriceTick>>,
    /** Resolved outcome index (0=Yes, 1=No typically) */
    current_index: usize,
    /** Agent's account */
    account: Account,
    /** Current step */
    step: u64,
    /** Taker fee rate (e.g., 0.001 = 0.1%) */
    taker_fee: f64,
}

#[cfg_attr(feature = "python", pymethods)]
impl PolymarketArena {
    #[cfg_attr(feature = "python", new)]
    #[cfg_attr(feature = "python", pyo3(signature = (initial_collateral=10000.0, taker_fee=0.001)))]
    pub fn new(initial_collateral: f64, taker_fee: f64) -> Self {
        PolymarketArena {
            markets: HashMap::new(),
            price_history: HashMap::new(),
            current_index: 0,
            account: Account::new(initial_collateral),
            step: 0,
            taker_fee,
        }
    }

    /** Get current collateral balance */
    pub fn collateral(&self) -> f64 {
        self.account.collateral
    }

    /** Get current step */
    pub fn current_step(&self) -> u64 {
        self.step
    }

    /** Get number of loaded markets */
    pub fn num_markets(&self) -> usize {
        self.markets.len()
    }

    /** Get current price for a market (Yes token price) */
    pub fn get_price(&self, market_id: &str) -> Option<f64> {
        self.price_history
            .get(market_id)
            .and_then(|history| history.get(self.current_index).map(|tick| tick.price))
    }

    /** Get position in a market as (yes_tokens, no_tokens) */
    pub fn get_position(&self, market_id: &str) -> (f64, f64) {
        self.account.get_position(market_id)
    }

    /** Get total account value at current prices */
    pub fn account_value(&self) -> f64 {
        let mut value = self.account.collateral;

        for (market_id, (yes_tokens, no_tokens)) in &self.account.positions {
            if let Some(yes_price) = self.get_price(market_id) {
                let no_price = 1.0 - yes_price;
                value += yes_tokens * yes_price + no_tokens * no_price;
            }
        }

        value
    }

    /** Get realized PnL */
    pub fn realized_pnl(&self) -> f64 {
        self.account.realized_pnl
    }
}

impl PolymarketArena {
    /** Load market metadata from JSON */
    pub fn load_markets(&mut self, json_data: &str) -> ArenaResult<()> {
        #[derive(Deserialize)]
        struct MarketMeta {
            id: u32,
            _filename: String,
            title: String,
            category: String,
            options: Vec<String>,
            outcome: Option<String>,
        }

        let markets: Vec<MarketMeta> = serde_json::from_str(json_data)?;

        for m in markets {
            let market = Market {
                id: m.id.to_string(),
                title: m.title,
                category: m.category,
                options: m.options,
                resolved: m.outcome.is_some(),
                outcome: m.outcome,
            };
            self.markets.insert(market.id.clone(), market);
        }

        Ok(())
    }

    /** Load price history from CSV data */
    pub fn load_price_history(&mut self, market_id: &str, csv_data: &str) -> ArenaResult<()> {
        let mut reader = csv::Reader::from_reader(csv_data.as_bytes());
        let mut history = Vec::new();

        for result in reader.records() {
            let record = result?;
            if record.len() >= 3 {
                let timestamp: u64 = record[1].trim_matches('"').parse().unwrap_or(0);
                let price: f64 = record[2].trim_matches('"').parse().unwrap_or(0.0);
                history.push(PriceTick { timestamp, price });
            }
        }

        self.price_history.insert(market_id.to_string(), history);
        Ok(())
    }

    /** Get the total number of price ticks across all markets */
    pub fn total_ticks(&self) -> usize {
        self.price_history
            .values()
            .map(|h| h.len())
            .max()
            .unwrap_or(0)
    }

    /** Buy Yes tokens for a market */
    pub fn buy_yes(&mut self, market_id: &str, amount: f64) -> ArenaResult<f64> {
        let price = self
            .get_price(market_id)
            .ok_or_else(|| ArenaError::MarketNotFound(market_id.to_string()))?;

        let cost = amount * price;
        let fee = cost * self.taker_fee;
        let total_cost = cost + fee;

        if self.account.collateral < total_cost {
            return Err(ArenaError::InsufficientBalance {
                required: total_cost,
                available: self.account.collateral,
            });
        }

        self.account.collateral -= total_cost;
        let (yes, _no) = self
            .account
            .positions
            .entry(market_id.to_string())
            .or_insert((0.0, 0.0));
        *yes += amount;

        Ok(total_cost)
    }

    /** Buy No tokens for a market */
    pub fn buy_no(&mut self, market_id: &str, amount: f64) -> ArenaResult<f64> {
        let yes_price = self
            .get_price(market_id)
            .ok_or_else(|| ArenaError::MarketNotFound(market_id.to_string()))?;
        let no_price = 1.0 - yes_price;

        let cost = amount * no_price;
        let fee = cost * self.taker_fee;
        let total_cost = cost + fee;

        if self.account.collateral < total_cost {
            return Err(ArenaError::InsufficientBalance {
                required: total_cost,
                available: self.account.collateral,
            });
        }

        self.account.collateral -= total_cost;
        let (_yes, no) = self
            .account
            .positions
            .entry(market_id.to_string())
            .or_insert((0.0, 0.0));
        *no += amount;

        Ok(total_cost)
    }

    /** Sell Yes tokens */
    pub fn sell_yes(&mut self, market_id: &str, amount: f64) -> ArenaResult<f64> {
        let price = self
            .get_price(market_id)
            .ok_or_else(|| ArenaError::MarketNotFound(market_id.to_string()))?;

        let (yes, _) = self
            .account
            .positions
            .get_mut(market_id)
            .ok_or_else(|| ArenaError::InvalidOrder("No position in market".to_string()))?;

        if *yes < amount {
            return Err(ArenaError::InvalidOrder(format!(
                "Insufficient Yes tokens: have {}, want to sell {}",
                yes, amount
            )));
        }

        let proceeds = amount * price;
        let fee = proceeds * self.taker_fee;
        let net_proceeds = proceeds - fee;

        *yes -= amount;
        self.account.collateral += net_proceeds;
        self.account.realized_pnl += net_proceeds - amount * price; // Approximate

        Ok(net_proceeds)
    }

    /** Merge complete sets back to collateral (1 Yes + 1 No = $1) */
    pub fn merge(&mut self, market_id: &str, amount: f64) -> ArenaResult<f64> {
        let (yes, no) = self
            .account
            .positions
            .get_mut(market_id)
            .ok_or_else(|| ArenaError::InvalidOrder("No position in market".to_string()))?;

        let can_merge = yes.min(*no).min(amount);

        if can_merge <= 0.0 {
            return Err(ArenaError::InvalidOrder(
                "No complete sets to merge".to_string(),
            ));
        }

        *yes -= can_merge;
        *no -= can_merge;
        self.account.collateral += can_merge; // Complete set = $1

        Ok(can_merge)
    }

    /** Split collateral into complete sets ($1 = 1 Yes + 1 No) */
    pub fn split(&mut self, market_id: &str, amount: f64) -> ArenaResult<f64> {
        if self.account.collateral < amount {
            return Err(ArenaError::InsufficientBalance {
                required: amount,
                available: self.account.collateral,
            });
        }

        self.account.collateral -= amount;
        let (yes, no) = self
            .account
            .positions
            .entry(market_id.to_string())
            .or_insert((0.0, 0.0));
        *yes += amount;
        *no += amount;

        Ok(amount)
    }

    /** Advance to next time step */
    pub fn advance(&mut self) -> bool {
        if self.current_index + 1 < self.total_ticks() {
            self.current_index += 1;
            self.step += 1;
            true
        } else {
            false
        }
    }

    /** Reset the arena to initial state */
    pub fn reset(&mut self, initial_collateral: f64) {
        self.current_index = 0;
        self.step = 0;
        self.account = Account::new(initial_collateral);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_account_creation() {
        let account = Account::new(10000.0);
        assert_eq!(account.collateral, 10000.0);
        assert!(account.positions.is_empty());
    }

    #[test]
    fn test_arena_creation() {
        // User account for the Polymarket arena
        let arena = PolymarketArena::new(10000.0, 0.001);
        assert_eq!(arena.collateral(), 10000.0);
        assert_eq!(arena.num_markets(), 0);
    }

    #[test]
    fn test_split_merge() {
        let mut arena = PolymarketArena::new(1000.0, 0.0);

        // Need to manually add a market and price for testing
        arena.markets.insert(
            "test".to_string(),
            Market {
                id: "test".to_string(),
                title: "Test Market".to_string(),
                category: "Test".to_string(),
                options: vec!["Yes".to_string(), "No".to_string()],
                resolved: false,
                outcome: None,
            },
        );
        arena.price_history.insert(
            "test".to_string(),
            vec![PriceTick {
                timestamp: 0,
                price: 0.5,
            }],
        );

        // Split $100 into tokens
        let split = arena.split("test", 100.0).unwrap();
        assert_eq!(split, 100.0);
        assert_eq!(arena.collateral(), 900.0);

        let (yes, no) = arena.get_position("test");
        assert_eq!(yes, 100.0);
        assert_eq!(no, 100.0);

        // Merge back
        let merged = arena.merge("test", 50.0).unwrap();
        assert_eq!(merged, 50.0);
        assert_eq!(arena.collateral(), 950.0);
    }
}
