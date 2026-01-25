/*!
 * Multi-Leg Spread Orders
 *
 * Defines structures for complex spread orders spanning multiple assets/books.
 * Supports Calendar Spreads, Vertical Spreads, Butterflies, and Custom Combinations.
 */

use crate::simulation::orderbook::{OrderBook, Side};
use serde::{Deserialize, Serialize};
use ts_rs::TS;

#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[ts(export)]
/// A single leg of a multi-leg order.
pub struct Leg {
    /// Asset symbol (must match a key in MultiAssetEnv orderbooks).
    pub asset: String,
    /// Side to execute on this leg.
    pub side: Side,
    /// Ratio of quantity relative to the spread quantity.
    /// E.g. For a Butterfly 1:2:1, outer legs have 1.0, inner has 2.0.
    pub ratio: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[ts(export)]
/// Type of spread strategy.
pub enum SpreadType {
    /// Purchase one asset, sell another (e.g., Calendar, Vertical).
    Calendar,
    Vertical,
    /// Low risk, limited profit strategy (e.g., Iron Condor, Butterfly).
    Butterfly,
    /// Custom user-defined combination.
    Custom,
}

#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[ts(export)]
/// A multi-leg spread order.
pub struct SpreadOrder {
    /// Strategy type.
    pub spread_type: SpreadType,
    /// List of legs definitions.
    pub legs: Vec<Leg>,
    /// Limit price for the *net* spread.
    /// Net Price = Sum(Leg Price * Ratio * (Side==Bid ? 1 : -1)) ??
    /// Convention: Debit is positive cost, Credit is negative cost.
    /// Limit: Max Debit (for buy) or Min Credit (for sell).
    pub limit_price: f64,
    /// Quantity of the complex order (units of the spread).
    pub quantity: f64,
}

impl SpreadOrder {
    /// Create a new generic spread order.
    pub fn new(legs: Vec<Leg>, limit_price: f64, quantity: f64) -> Self {
        Self {
            spread_type: SpreadType::Custom,
            legs,
            limit_price,
            quantity,
        }
    }

    /// Create a standard vertical spread (Buy Low Strike, Sell High Strike).
    pub fn new_vertical_spread(
        long_asset: String,
        short_asset: String,
        price_diff: f64,
        quantity: f64,
    ) -> Self {
        Self {
            spread_type: SpreadType::Vertical,
            legs: vec![
                Leg {
                    asset: long_asset,
                    side: Side::Bid,
                    ratio: 1.0,
                },
                Leg {
                    asset: short_asset,
                    side: Side::Ask,
                    ratio: 1.0,
                },
            ],
            limit_price: price_diff,
            quantity,
        }
    }

    /// Create a standard butterfly spread (Buy 1 Low, Sell 2 Mid, Buy 1 High).
    pub fn new_butterfly(
        low_asset: String,
        mid_asset: String,
        high_asset: String,
        limit_price: f64,
        quantity: f64,
    ) -> Self {
        Self {
            spread_type: SpreadType::Butterfly,
            legs: vec![
                Leg {
                    asset: low_asset,
                    side: Side::Bid,
                    ratio: 1.0,
                },
                Leg {
                    asset: mid_asset,
                    side: Side::Ask,
                    ratio: 2.0,
                },
                Leg {
                    asset: high_asset,
                    side: Side::Bid,
                    ratio: 1.0,
                },
            ],
            limit_price,
            quantity,
        }
    }

    /// Check if the spread order can execute against current market prices (marketable).
    /// Returns true if the net cost of taking liquidity across all legs meets the limit price.
    ///
    /// Note: This assumes "Fill or Kill" logic for the spread (atomic).
    pub fn can_execute(&self, books: &std::collections::HashMap<String, OrderBook>) -> bool {
        let mut net_cost = 0.0;

        for leg in &self.legs {
            let book = match books.get(&leg.asset) {
                Some(b) => b,
                None => return false, // Asset not found, cannot execute
            };

            // To buy spread leg (Bid), we hit the Ask.
            // To sell spread leg (Ask), we hit the Bid.
            let target_price = match leg.side {
                Side::Bid => book.best_ask(), // We buy, so we pay Ask
                Side::Ask => book.best_bid(), // We sell, so we get Bid
            };

            let p = match target_price {
                Some(p) => p,
                None => return false, // No liquidity
            };

            // Cost convention: Buying is positive cost, Selling is negative cost (revenue)
            let leg_cost = match leg.side {
                Side::Bid => p * leg.ratio,
                Side::Ask => -p * leg.ratio,
            };

            net_cost += leg_cost;
        }

        // Limit condition:
        // If we are paying (Net > 0), we want Cost <= Limit.
        // If we are receiving (Net < 0), we want Proceeds >= Limit (Cost <= Limit still holds if Limit is negative revenue).
        // Convention: limit_price is the max we are willing to pay (Debit).
        // If negative, it's the min we want to receive (Credit).

        net_cost <= self.limit_price
    }
}
