//! nglab - High-Performance RL Arena for Financial Trading
//!
//! This crate provides a Rust-based simulation engine for reinforcement learning
//! in financial markets, with Python bindings via PyO3.

#[cfg(feature = "python")]
use pyo3::prelude::*;

pub mod errors;
pub mod models;
pub mod simulator;
pub mod utils;
pub mod web;

pub use errors::{ArenaError, ArenaResult};

/// Arena - The main simulation environment
#[cfg_attr(feature = "python", pyclass)]
pub struct Arena {
    step_count: u64,
}

#[cfg_attr(feature = "python", pymethods)]
impl Arena {
    #[cfg_attr(feature = "python", new)]
    pub fn new() -> Self {
        Arena { step_count: 0 }
    }

    /// Get the current step count
    pub fn step_count(&self) -> u64 {
        self.step_count
    }
}

impl Default for Arena {
    fn default() -> Self {
        Self::new()
    }
}

/// Python module entry point
#[cfg(feature = "python")]
#[pymodule]
fn _nglab(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Arena>()?;
    m.add_class::<simulator::orderbook::OrderBook>()?;
    m.add_class::<simulator::polymarket::PolymarketArena>()?;
    m.add_class::<simulator::gym::TradingEnv>()?;
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
