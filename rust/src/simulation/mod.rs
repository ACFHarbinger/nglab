//! Core simulation engines and trading environments.
//!
//! This module provides the high-performance backtesting and simulation
//! infrastructure, including order books, reinforcement learning gym environments,
//! multi-asset trading simulators, and risk management systems.

/// Circuit breaker for volatility protection.
pub mod circuit_breaker;
/// Reinforcement learning environment (Gym-compatible).
pub mod gym;
/// Market making logic and spread management.
pub mod market_maker;
/// Multi-agent simulation framework.
pub mod multi_agent;
/// Multi-asset market simulation.
pub mod multi_asset;
/// Options market simulation.
pub mod options;
/// Central Limit Order Book (CLOB).
pub mod orderbook;
pub mod paper_trading;
/// Polymarket-specific simulation logic.
pub mod polymarket;
/// Risk management and position sizing.
pub mod risk;
/// Scenario analysis and stress testing.
pub mod scenarios;
pub mod spreads;
