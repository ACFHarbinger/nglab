use crate::simulation::orderbook::{OrderBook, Side};
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

/**
 * Trait for automated trading agents in the simulation.
 */
pub trait Agent: Debug + Send + Sync {
    /// Unique identifier for the agent
    fn id(&self) -> String;
    /// Generate actions based on the current order book state
    fn act(&mut self, orderbook: &OrderBook, timestamp: u64) -> Vec<AgentAction>;
}

/**
 * Actions that an agent can take.
 */
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgentAction {
    /// Submit a limit order
    SubmitLimit {
        /// Price of the limit order
        price: f64,
        /// Quantity to trade
        quantity: f64,
        /// Buy or Sell
        side: Side,
    },
    /// Submit a market order
    SubmitMarket {
        /// Quantity to trade
        quantity: f64,
        /// Buy or Sell
        side: Side,
    },
    /// Cancel an existing order
    CancelOrder {
        /// ID of order to cancel
        order_id: u64,
    },
}

/**
 * Noise Agent - Provides baseline liquidity with random orders around the mid price.
 */
#[derive(Debug, Clone)]
pub struct NoiseAgent {
    id: String,
    rng_seed: u64,
}

impl NoiseAgent {
    /// Create a new Noise Agent.
    pub fn new(id: &str, seed: u64) -> Self {
        Self {
            id: id.to_string(),
            rng_seed: seed,
        }
    }
}

impl Agent for NoiseAgent {
    fn id(&self) -> String {
        self.id.clone()
    }

    fn act(&mut self, orderbook: &OrderBook, _timestamp: u64) -> Vec<AgentAction> {
        let mut actions = Vec::new();
        if let Some(mid) = orderbook.mid_price() {
            // Very simple noise: buy or sell small amount at mid price +/- small offset
            // In a real implementation, we'd use proper RNG here.
            // Using a simple deterministic "randomness" for now based on seed/mid.
            let offset = (self.rng_seed as f64 % 10.0) / 100.0; // 0.0 to 0.1

            if self.rng_seed % 2 == 0 {
                actions.push(AgentAction::SubmitLimit {
                    price: mid - offset,
                    quantity: 1.0,
                    side: Side::Bid,
                });
            } else {
                actions.push(AgentAction::SubmitLimit {
                    price: mid + offset,
                    quantity: 1.0,
                    side: Side::Ask,
                });
            }
            self.rng_seed = self.rng_seed.wrapping_add(1);
        }
        actions
    }
}

/**
 * Momentum Agent - Follows price trends.
 */
#[derive(Debug, Clone)]
pub struct MomentumAgent {
    id: String,
    last_price: Option<f64>,
}

impl MomentumAgent {
    /// Create a new Momentum Agent.
    pub fn new(id: &str) -> Self {
        Self {
            id: id.to_string(),
            last_price: None,
        }
    }
}

impl Agent for MomentumAgent {
    fn id(&self) -> String {
        self.id.clone()
    }

    fn act(&mut self, orderbook: &OrderBook, _timestamp: u64) -> Vec<AgentAction> {
        let mut actions = Vec::new();
        if let Some(mid) = orderbook.mid_price() {
            if let Some(prev) = self.last_price {
                if mid > prev {
                    // Upward trend - Buy
                    actions.push(AgentAction::SubmitMarket {
                        quantity: 1.0,
                        side: Side::Bid,
                    });
                } else if mid < prev {
                    // Downward trend - Sell
                    actions.push(AgentAction::SubmitMarket {
                        quantity: 1.0,
                        side: Side::Ask,
                    });
                }
            }
            self.last_price = Some(mid);
        }
        actions
    }
}

/**
 * Manager for agent population.
 */
pub struct AgentManager {
    agents: Vec<Box<dyn Agent>>,
}

impl AgentManager {
    /// Create a new Agent Manager.
    pub fn new() -> Self {
        Self { agents: Vec::new() }
    }

    /// Add an agent to the manager.
    pub fn add_agent(&mut self, agent: Box<dyn Agent>) {
        self.agents.push(agent);
    }

    /// Step the simulation for all agents.
    pub fn step(&mut self, orderbook: &mut OrderBook, timestamp: u64) -> Vec<TradeRecord> {
        let mut all_trades = Vec::new();
        let mut actions = Vec::new();

        // 1. Collect actions from all agents
        for agent in &mut self.agents {
            actions.extend(agent.act(orderbook, timestamp));
        }

        // 2. Execute actions
        for action in actions {
            match action {
                AgentAction::SubmitLimit {
                    price,
                    quantity,
                    side,
                } => {
                    if let Ok((_id, trades)) = orderbook.submit_limit_order(price, quantity, side) {
                        for t in trades {
                            all_trades.push(TradeRecord::from_trade(t, &self.agents[0].id()));
                            // Simple attribution for now
                        }
                    }
                }
                AgentAction::SubmitMarket { quantity, side } => {
                    if let Ok((_id, trades)) = orderbook.submit_market_order(quantity, side) {
                        for t in trades {
                            all_trades.push(TradeRecord::from_trade(t, &self.agents[0].id()));
                        }
                    }
                }
                AgentAction::CancelOrder { order_id } => {
                    orderbook.cancel_order(order_id);
                }
            }
        }
        all_trades
    }
}

/**
 * Simplified trade record for agent performance tracking.
 */
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeRecord {
    /// ID of the taker order
    pub order_id: u64,
    /// Price of the trade
    pub price: f64,
    /// Quantity traded
    pub quantity: f64,
    /// Side of the taker
    pub side: Side,
    /// Timestamp of the trade
    pub timestamp: u64,
    /// ID of the agent who took the action
    pub agent_id: String,
}

impl TradeRecord {
    fn from_trade(trade: crate::simulation::orderbook::Trade, agent_id: &str) -> Self {
        Self {
            order_id: trade.taker_order_id,
            price: trade.price,
            quantity: trade.quantity,
            side: trade.side,
            timestamp: trade.timestamp,
            agent_id: agent_id.to_string(),
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulation::orderbook::OrderBook;

    #[test]
    fn test_noise_agent() {
        let mut book = OrderBook::new();
        book.submit_limit_order(100.0, 10.0, Side::Bid).unwrap();
        book.submit_limit_order(102.0, 10.0, Side::Ask).unwrap();

        // Mid price is 101.0
        let mut agent = NoiseAgent::new("noise_1", 42);
        let actions = agent.act(&book, 100);

        assert_eq!(actions.len(), 1);
        match &actions[0] {
            AgentAction::SubmitLimit { price, .. } => {
                assert!(*price >= 100.0 && *price <= 102.0);
            }
            _ => panic!("Expected SubmitLimit action"),
        }
    }

    #[test]
    fn test_agent_manager() {
        let mut book = OrderBook::new();
        book.submit_limit_order(100.0, 10.0, Side::Bid).unwrap();
        book.submit_limit_order(102.0, 10.0, Side::Ask).unwrap();

        let mut manager = AgentManager::new();
        manager.add_agent(Box::new(NoiseAgent::new("noise_1", 42)));
        manager.add_agent(Box::new(MomentumAgent::new("mom_1")));

        let trades = manager.step(&mut book, 100);
        // Step 1: Noise agent submits a limit order (no trade)
        // Momentum agent observes mid=101, no prev price (no trade)
        assert_eq!(trades.len(), 0);

        // Change mid price to trigger momentum
        book.submit_limit_order(103.0, 10.0, Side::Ask).unwrap();

        let _trades = manager.step(&mut book, 101);
        // Momentum agent sees mid change, might submit market order
        // Noise agent continues.
        // Even if 0 trades, we verify it doesn't panic.
    }
}
