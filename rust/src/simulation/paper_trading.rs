//! Paper trading account and state management.
//!
//! Provides virtual balance tracking, position monitoring, and persistent
//! state for the Paper Trading mode.

use crate::simulation::orderbook::Order;
use serde::{Deserialize, Serialize};
use serde_json;
use std::collections::HashMap;
use std::fs;
use std::io;

/// Represents a Paper Trading account state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaperAccount {
    /// Available cash balance.
    pub balance: f64,
    /// Total equity (balance + unrealized P&L).
    pub equity: f64,
    /// Open positions: Map of asset symbols to quantities.
    pub positions: HashMap<String, f64>,
    /// Active orders.
    pub orders: Vec<Order>,
}

impl Default for PaperAccount {
    fn default() -> Self {
        Self {
            balance: 100_000.0, // Default virtual starting balance
            equity: 100_000.0,
            positions: HashMap::new(),
            orders: Vec::new(),
        }
    }
}

impl PaperAccount {
    /// Create a new paper account with a specific starting balance.
    pub fn new(initial_balance: f64) -> Self {
        Self {
            balance: initial_balance,
            equity: initial_balance,
            positions: HashMap::new(),
            orders: Vec::new(),
        }
    }

    /// Reset the account to its initial state.
    pub fn reset(&mut self, initial_balance: f64) {
        self.balance = initial_balance;
        self.equity = initial_balance;
        self.positions.clear();
        self.orders.clear();
    }

    /// Update the account's total equity based on current market prices.
    pub fn update_equity(&mut self, market_prices: &HashMap<String, f64>) {
        let mut unrealized_pnl = 0.0;
        for (symbol, qty) in &self.positions {
            if let Some(price) = market_prices.get(symbol) {
                unrealized_pnl += qty * price;
            }
        }
        self.equity = self.balance + unrealized_pnl;
    }

    /// Submit an order to the paper account.
    pub fn submit_order(&mut self, mut order: Order) -> u64 {
        let id = order.id;
        self.orders.push(order);
        id
    }

    /// Apply a fill to the account.
    pub fn apply_fill(
        &mut self,
        symbol: &str,
        qty: f64,
        price: f64,
        side: crate::simulation::orderbook::Side,
    ) {
        let cost = qty * price;
        match side {
            crate::simulation::orderbook::Side::Bid => {
                self.balance -= cost;
                *self.positions.entry(symbol.to_string()).or_insert(0.0) += qty;
            }
            crate::simulation::orderbook::Side::Ask => {
                self.balance += cost;
                *self.positions.entry(symbol.to_string()).or_insert(0.0) -= qty;
            }
        }
    }

    /// Check for fills against the current state of an order book.
    pub fn check_fills(&mut self, symbol: &str, book: &crate::simulation::orderbook::OrderBook) {
        let mut f_ids = Vec::new();
        let mut fills = Vec::new();

        for (i, order) in self.orders.iter_mut().enumerate() {
            match order.side {
                crate::simulation::orderbook::Side::Bid => {
                    if let Some(ask) = book.best_ask() {
                        if order.price >= ask {
                            // Simplified fill: execute at best ask
                            let fill_qty = order.remaining();
                            fills.push((symbol.to_string(), fill_qty, ask, order.side));
                            order.filled += fill_qty;
                            f_ids.push(i);
                        }
                    }
                }
                crate::simulation::orderbook::Side::Ask => {
                    if let Some(bid) = book.best_bid() {
                        if order.price <= bid {
                            let fill_qty = order.remaining();
                            fills.push((symbol.to_string(), fill_qty, bid, order.side));
                            order.filled += fill_qty;
                            f_ids.push(i);
                        }
                    }
                }
            }
        }

        for (sym, qty, price, side) in fills {
            self.apply_fill(&sym, qty, price, side);
        }

        // Remove filled orders (in reverse to preserve indices)
        f_ids.sort_by(|a, b| b.cmp(a));
        for i in f_ids {
            self.orders.remove(i);
        }
    }

    /// Save the account to a file.
    pub fn save(&self, path: &str) -> io::Result<()> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        fs::write(path, json)?;
        Ok(())
    }

    /// Load the account from a file.
    pub fn load(path: &str) -> io::Result<Self> {
        let json = fs::read_to_string(path)?;
        let account: Self =
            serde_json::from_str(&json).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        Ok(account)
    }
}
