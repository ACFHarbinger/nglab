//! Core simulation engines and trading environments.
//!
//! This module provides the high-performance backtesting and simulation
//! infrastructure, including order books, reinforcement learning gym environments,
//! multi-asset trading simulators, and risk management systems.

/// Reinforcement learning environment (Gym-compatible).
pub mod gym;
/// Multi-asset market simulation.
pub mod multi_asset;
/// Central Limit Order Book (CLOB).
pub mod orderbook;
/// Polymarket-specific simulation logic.
pub mod polymarket;
/// Risk management and position sizing.
pub mod risk;
