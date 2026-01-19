/*!
 * nglab - High-Performance RL Arena for Financial Trading
 *
 * This crate provides a Rust-based simulation engine for reinforcement learning
 * in financial markets, exposing a Gymnasium-compatible interface to Python via PyO3.
 *
 * # Core Components
 *
 * - **Arena**: The central simulation controller that manages the environment lifecycle,
 *   time-stepping, and agent interactions.
 * - **OrderBook**: A high-performance Central Limit Order Book (CLOB) implementation
 *   supporting price-time priority, limit/market orders, and realistic fill simulation.
 * - **TradingEnv**: An OpenAI Gym-compatible environment wrapper that handles observation
 *   normalization, reward calculation, and action space mapping.
 *
 * # Architecture
 *
 * The system is designed as a hybrid Rust/Python application:
 * - **Rust Layer**: Handles the computationally intensive simulation loop, order matching,
 *   and state management. It aims for zero-allocation in the hot path.
 * - **Python Layer**: Consumes the Rust primitives via PyO3, providing a familiar API for
 *   ML researchers using PyTorch, TensorFlow, or JAX.
 *
 * # Example Usage
 *
 * While primarily designed to be used from Python, the core components can be used natively in Rust:
 *
 * ```rust,no_run
 * use nglab::simulation::TradingEnv;
 *
 * // Initialize environment with starting capital and transaction costs
 * let mut env = TradingEnv::new(10_000.0, 0.001, 10, 100, false);
 *
 * // Reset environment
 * let initial_obs = env.reset_rs();
 *
 * // Step through simulation
 * let action = 1; // e.g., Buy
 * let (obs, reward, terminated, truncated, info) = env.step_rs(action);
 *
 * println!("Reward: {}, Portfolio: {}", reward, info.portfolio_value);
 * ```
 */

#[cfg(feature = "python")]
use pyo3::prelude::*;

pub mod config;
pub mod errors;
pub mod health;
pub mod logging;
pub mod models;
pub mod moon;
pub mod secret;
pub mod simulation;
pub mod utils;
pub mod web;

pub use errors::{ArenaError, ArenaResult};

/**
 * Arena - The main simulation environment.
 *
 * Manages the state and stepping of the trading simulation.
 */
#[cfg_attr(feature = "python", pyclass)]
pub struct Arena {
    /** Total number of steps taken in the arena */
    step_count: u64,
}

// =========================================================================
// Python Bindings Implementation
// =========================================================================

#[cfg(feature = "python")]
#[pymethods]
impl Arena {
    #[new]
    pub fn new_py() -> Self {
        Self::new()
    }

    #[pyo3(name = "step_count")]
    pub fn step_count_py(&self) -> u64 {
        self.step_count()
    }
}

// =========================================================================
// Pure Rust Implementation
// =========================================================================

impl Arena {
    /**
     * Create a new instance of the Arena with default state.
     */
    pub fn new() -> Self {
        Arena { step_count: 0 }
    }

    /**
     * Get the current step count of the simulation.
     *
     * Returns the number of steps that have been executed.
     */
    pub fn step_count(&self) -> u64 {
        self.step_count
    }
}

impl Default for Arena {
    fn default() -> Self {
        Self::new()
    }
}

/**
 * Python module entry point for the `nglab` extension.
 *
 * Exposes Rust classes and methods to the Python environment.
 */
#[cfg(feature = "python")]
#[pymodule]
fn _nglab(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Arena>()?;
    m.add_class::<simulation::orderbook::OrderBook>()?;
    m.add_class::<simulation::polymarket::PolymarketArena>()?;
    m.add_class::<simulation::gym::TradingEnv>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arena_creation() {
        let arena = Arena::new();
        assert_eq!(arena.step_count(), 0);
    }
}
