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
    /// Error related to order book logic or consistency.
    #[error("Order book error: {0}")]
    OrderBook(String),

    /// Error for invalid order parameters (price, quantity).
    #[error("Invalid order: {0}")]
    InvalidOrder(String),

    /// Error when a trade cannot be collateralized.
    #[error("Insufficient balance: required {required}, available {available}")]
    InsufficientBalance {
        /// The amount of collateral required.
        required: f64,
        /// The available collateral amount.
        available: f64,
    },

    /// Price validation fail.
    #[error("Invalid price: {0}")]
    InvalidPrice(f64),

    /// Quantity validation fail.
    #[error("Invalid quantity: {0}")]
    InvalidQuantity(f64),

    /// Market or asset identifier not found in arena.
    #[error("Market not found: {0}")]
    MarketNotFound(String),

    /// Failure during time-series or parameter loading.
    #[error("Data loading error: {0}")]
    DataLoading(String),

    /// General I/O failure.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Failure processing CSV data files.
    #[error("CSV error: {0}")]
    Csv(#[from] csv::Error),

    /// failure processing JSON data.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// Network or API communication failure.
    #[error("Network error: {0}")]
    Network(#[from] reqwest::Error),

    /// Error returned from or through the Python bridge.
    #[error("Python error: {0}")]
    Python(String),

    /// Failure within a financial or forecasting model.
    #[error("Model error: {0}")]
    ModelError(String),

    /// Numerical instability or calculation failure (NaN, Infinity).
    #[error("Numerical error: {0}")]
    NumericalError(String),

    /// Unexpected internal state or logic failure.
    #[error("Internal error: {0}")]
    InternalError(String),
    /// Simulation parameter validation failure.
    #[error("Validation error: {0}")]
    ValidationError(String),
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
