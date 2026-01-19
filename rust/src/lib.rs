/*!
 * nglab - High-Performance RL Arena for Financial Trading
 *
 * This crate provides a Rust-based simulation engine for reinforcement learning
 * in financial markets, with Python bindings via PyO3.
 *
 * Core components include the `Arena`, `OrderBook`, and various market simulators.
 */

#[cfg(feature = "python")]
use pyo3::prelude::*;

pub mod errors;
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
