//! nglab - High-Performance RL Arena for Financial Trading
//!
//! This crate provides a Rust-based simulation engine for reinforcement learning
//! in financial markets, with Python bindings via PyO3.

use pyo3::prelude::*;

pub mod error;
pub mod gym;
pub mod orderbook;
pub mod polymarket;

pub use error::{ArenaError, ArenaResult};

/// Arena - The main simulation environment
#[pyclass]
pub struct Arena {
    step_count: u64,
}

#[pymethods]
impl Arena {
    #[new]
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
#[pymodule]
fn nglab(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Arena>()?;
    m.add_class::<orderbook::OrderBook>()?;
    m.add_class::<polymarket::PolymarketArena>()?;
    m.add_class::<gym::TradingEnv>()?;
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
