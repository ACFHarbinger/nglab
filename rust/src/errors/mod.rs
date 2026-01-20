/*!
 * Error types for the arena.
 *
 * Standardizes error handling and result types across
 * the simulation and model components.
 */

use thiserror::Error;

/**
 * Comprehensive error enumeration for all arena and simulation failures.
 */
#[derive(Error, Debug)]
pub enum ArenaError {
    #[error("Order book error: {0}")]
    OrderBook(String),

    #[error("Invalid order: {0}")]
    InvalidOrder(String),

    #[error("Insufficient balance: required {required}, available {available}")]
    InsufficientBalance { required: f64, available: f64 },

    #[error("Invalid price: {0}")]
    InvalidPrice(f64),

    #[error("Invalid quantity: {0}")]
    InvalidQuantity(f64),

    #[error("Market not found: {0}")]
    MarketNotFound(String),

    #[error("Data loading error: {0}")]
    DataLoading(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("CSV error: {0}")]
    Csv(#[from] csv::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Network error: {0}")]
    Network(#[from] reqwest::Error),

    #[error("Python error: {0}")]
    Python(String),
}

#[cfg(feature = "python")]
use pyo3::{exceptions::PyRuntimeError, PyErr};

#[cfg(feature = "python")]
impl From<ArenaError> for PyErr {
    fn from(err: ArenaError) -> PyErr {
        PyRuntimeError::new_err(err.to_string())
    }
}

/**
 * Specialized Result type for arena operations.
 */
pub type ArenaResult<T> = Result<T, ArenaError>;
