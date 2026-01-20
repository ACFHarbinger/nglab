/*!
 * Gymnasium-compatible trading environment
 *
 * Implements the standard RL interface (reset, step) with:
 * - Discrete and continuous action spaces
 * - Normalized observations via zero-copy numpy arrays
 * - Risk-adjusted reward functions
 */

use crate::errors::{ArenaError, ArenaResult};
use crate::simulation::orderbook::{OrderBook, Side};
use crate::simulation::risk::{RiskManager, RiskStatus};
use crate::utils::math::{safe_div, SafeFloat};
#[cfg(feature = "python")]
use numpy::{PyArray2, ToPyArray};
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyDict;
use rand::rngs::StdRng;
use rand::SeedableRng;
use smallvec::SmallVec;
#[cfg(feature = "python")]
use tracing::info;
use tracing::{debug, instrument};

/**
 * Action space types for the agent.
 */
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActionType {
    /** Hold current position */
    Hold = 0,
    /** Buy (long) */
    Buy = 1,
    /** Sell (short) */
    Sell = 2,
}

impl From<i32> for ActionType {
    fn from(value: i32) -> Self {
        match value {
            0 => ActionType::Hold,
            1 => ActionType::Buy,
            2 => ActionType::Sell,
            _ => ActionType::Hold,
        }
    }
}

/**
 * Step result returned to high-level callers (e.g., Python).
 *
 * Bundles the new observation, the reward received for the action,
 * and flags indicating if the episode has finished.
 *
 * # Fields
 *
 * * `observation` - Vector of feature values for the new state.
 * * `reward` - Scaled reward value (e.g., risk-adjusted return).
 * * `terminated` - True if the agent reached a terminal state (e.g., bankruptcy).
 * * `truncated` - True if the simulation was cut short (e.g., time limit).
 * * `info` - Additional diagnostic metadata.
 */
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StepResult {
    pub observation: Vec<f64>,
    pub reward: f64,
    pub terminated: bool,
    pub truncated: bool,
    pub info: StepInfo,
}

/**
 * Additional metrics and diagnostics for each step.
 */
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct StepInfo {
    pub portfolio_value: f64,
    pub position: f64,
    pub cash: f64,
    pub sharpe_ratio: f64,
    pub total_steps: u64,
}

/// Pre-allocated observation buffer for zero-allocation hot path.
/// Uses a fixed-size array internally to avoid heap allocations.
pub struct ObservationBuffer {
    data: Vec<f64>,
    shape: (usize, usize),
}

impl ObservationBuffer {
    pub fn new(features: usize, lookback: usize) -> Self {
        ObservationBuffer {
            data: vec![0.0; features * lookback],
            shape: (lookback, features),
        }
    }

    /// Update a single row of observations. Zero-copy when possible.
    #[inline]
    pub fn update(&mut self, row: usize, values: &[f64]) {
        let start = row * self.shape.1;
        let end = start + values.len().min(self.shape.1);
        self.data[start..end].copy_from_slice(&values[..end - start]);
    }

    /// Get the underlying data as a slice.
    #[inline]
    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }

    /// Reset buffer to zeros.
    #[inline]
    pub fn reset(&mut self) {
        self.data.fill(0.0);
    }

    /// Clone data into a new Vec (for Python interop).
    pub fn to_vec(&self) -> Vec<f64> {
        self.data.clone()
    }
}

/// Stack-allocated returns history (avoids heap for typical window sizes).
type ReturnsHistory = SmallVec<[f64; 64]>;

/**
 * Trading environment with Gymnasium interface.
 *
 * Wraps market simulation logic into a standard RL format suitable for training agents.
 * It manages the interaction between the agent (action) and the market (order book),
 * computing rewards and updating state variables.
 *
 * # Features
 *
 * - **Zero-Copy Observations**: Leverages `numpy` crate for efficient data transfer to Python.
 * - **Configurable Rewards**: Supports various reward schemes (PnL, Sharpe Ratio, etc.).
 * - **Market Simulation**: Integrated with `OrderBook` for realistic execution.
 *
 * # Fields
 *
 * * `orderbook` - The central limit order book managing market liquidity.
 * * `position` - Current net position (positive for long, negative = short).
 * * `cash` - Available cash balance for trading.
 * * `portfolio_value` - Total account equity (Cash + Mark-to-Market Position).
 * * `step_count` - Current time step in the episode.
 */
#[cfg_attr(feature = "python", pyclass)]
pub struct TradingEnv {
    /** Order book for simulation */
    orderbook: OrderBook,
    /** Price history for backtesting */
    prices: Vec<f64>,
    /** Current step index */
    current_step: usize,
    /** Current position (positive = long, negative = short) */
    position: f64,
    /** Current cash balance */
    cash: f64,
    /** Initial capital */
    initial_capital: f64,
    /** Transaction cost rate */
    transaction_cost: f64,
    /** Lookback window for observations */
    lookback: usize,
    /** Number of features per timestep */
    num_features: usize,
    /** Returns history for Sharpe calculation (stack-allocated for small windows) */
    returns: ReturnsHistory,
    /** Previous portfolio value for return calculation */
    prev_portfolio_value: f64,
    /** Maximum steps before truncation */
    max_steps: usize,
    /** Total steps taken */
    total_steps: u64,
    /** Rerun logger for visualization */
    logger: crate::utils::visualizer::RerunLogger,
    /** Pre-allocated observation buffer */
    obs_buffer: ObservationBuffer,
    /** Random number generator for reproducibility */
    rng: StdRng,
    /** Risk manager for monitoring and enforcing risk limits */
    risk_manager: RiskManager,
}

// =========================================================================
// Python Bindings Implementation
// =========================================================================

#[cfg(feature = "python")]
#[pymethods]
impl TradingEnv {
    #[new]
    #[pyo3(signature = (initial_capital=10000.0, transaction_cost=0.001, lookback=30, max_steps=1000, enable_logging=true, seed=None))]
    pub fn new_py(
        initial_capital: f64,
        transaction_cost: f64,
        lookback: usize,
        max_steps: usize,
        enable_logging: bool,
        seed: Option<u64>,
    ) -> Self {
        Self::new(
            initial_capital,
            transaction_cost,
            lookback,
            max_steps,
            enable_logging,
            seed,
        )
    }

    /** Load historical price data */
    #[pyo3(name = "load_prices")]
    pub fn load_prices_py(&mut self, prices: Vec<f64>) {
        self.load_prices(prices);
    }

    /** Reset the environment */
    #[pyo3(signature = (seed=None, options=None))]
    #[instrument(skip(self, py), fields(step = self.total_steps))]
    pub fn reset<'py>(
        &mut self,
        py: Python<'py>,
        seed: Option<u64>,
        options: Option<Py<PyAny>>,
    ) -> PyResult<(Bound<'py, PyArray2<f64>>, Py<PyAny>)> {
        if let Some(s) = seed {
            info!("Resetting environment with seed: {}", s);
            self.set_seed(s);
        } else {
            info!("Resetting environment with existing seed");
        }
        self.reset_rs();

        let obs = self.get_observation(py)?;
        let info = PyDict::new(py).into();
        Ok((obs, info))
    }

    /** Take a step in the environment */
    #[allow(clippy::type_complexity)]
    #[instrument(skip(self, py), fields(step = self.total_steps, action = action))]
    pub fn step<'py>(
        &mut self,
        py: Python<'py>,
        action: i32,
    ) -> PyResult<(Bound<'py, PyArray2<f64>>, f64, bool, bool, Py<PyAny>)> {
        let action_type = ActionType::from(action);
        let lookback = self.lookback;
        let num_features = self.num_features;

        debug!("Executing action: {:?}", action_type);

        let trade_cost = self.execute_action(action_type)?;
        self.current_step += 1;
        self.total_steps += 1;

        let portfolio_value = self.portfolio_value();
        let returns = safe_div(
            portfolio_value - self.prev_portfolio_value,
            self.prev_portfolio_value,
            0.0,
        );
        self.returns.push(returns);
        self.prev_portfolio_value = portfolio_value;

        // Update risk manager
        self.risk_manager.update(portfolio_value);

        let reward = self.calculate_reward(returns, trade_cost);
        let terminated =
            portfolio_value <= 0.0 || self.current_step >= self.prices.len().saturating_sub(1);
        let truncated = self.current_step.saturating_sub(self.lookback) >= self.max_steps;

        let obs_data = self.generate_observation_data();

        let step_info = StepInfo {
            portfolio_value,
            position: self.position,
            cash: self.cash,
            sharpe_ratio: self.calculate_sharpe(30),
            total_steps: self.total_steps,
        };

        self.logger
            .log_step(self.total_steps, &self.orderbook, portfolio_value);

        // Build info dict
        let dict = PyDict::new(py);
        dict.set_item("portfolio_value", step_info.portfolio_value)?;
        dict.set_item("position", step_info.position)?;
        dict.set_item("cash", step_info.cash)?;
        dict.set_item("sharpe_ratio", step_info.sharpe_ratio)?;
        dict.set_item("total_steps", step_info.total_steps)?;
        let info = dict.into();

        // Create numpy array
        let obs_array = ndarray::Array2::from_shape_vec((lookback, num_features), obs_data)
            .map_err(|e| ArenaError::DataLoading(format!("Invalid observation shape: {}", e)))?;
        let obs = obs_array.to_pyarray(py);

        Ok((obs, reward, terminated, truncated, info))
    }

    /** Get current observation as numpy array */
    pub fn get_observation<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let obs_data = self.generate_observation_data();

        // Reshape to (lookback, features)
        let array = ndarray::Array2::from_shape_vec((self.lookback, self.num_features), obs_data)
            .map_err(|e| ArenaError::DataLoading(format!("Invalid shape: {}", e)))?;

        Ok(array.to_pyarray(py))
    }
}

// =========================================================================
// Pure Rust Implementation
// =========================================================================

impl TradingEnv {
    /**
     * Create a new trading environment.
     *
     * # Arguments
     *
     * * `initial_capital` - Starting cash balance in USDC.
     * * `transaction_cost` - Taker fee rate (e.g., 0.001 for 0.1%).
     * * `lookback` - Number of historical ticks to include in observations.
     * * `max_steps` - Maximum duration of a single episode.
     * * `enable_logging` - Whether to stream telemetry to Rerun.
     */
    pub fn new(
        initial_capital: f64,
        transaction_cost: f64,
        lookback: usize,
        max_steps: usize,
        enable_logging: bool,
        seed: Option<u64>,
    ) -> Self {
        let num_features = 6; // price, return, volume, imbalance, position, cash
        let rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => {
                // Use rand::rng() to generate a random seed (rand 0.9 API)
                use rand::Rng;
                let random_seed = rand::rng().random::<u64>();
                StdRng::seed_from_u64(random_seed)
            }
        };
        TradingEnv {
            orderbook: OrderBook::new(),
            prices: Vec::new(),
            current_step: 0, // Will be set to lookback on reset
            position: 0.0,
            cash: initial_capital,
            initial_capital,
            transaction_cost,
            lookback,
            num_features,
            returns: SmallVec::new(),
            prev_portfolio_value: initial_capital,
            max_steps,
            total_steps: 0,
            logger: crate::utils::visualizer::RerunLogger::new(enable_logging),
            obs_buffer: ObservationBuffer::new(num_features, lookback),
            rng,
            risk_manager: RiskManager::with_defaults(initial_capital),
        }
    }

    /**
     * Load historical price data for backtesting.
     *
     * @param prices A vector of price points.
     */
    pub fn load_prices(&mut self, prices: Vec<f64>) {
        self.prices = prices;
    }

    /** Get observation space shape */
    pub fn observation_shape(&self) -> (usize, usize) {
        (self.lookback, self.num_features)
    }

    /** Get action space size (discrete: 0=hold, 1=buy, 2=sell) */
    pub fn action_space_size(&self) -> usize {
        3
    }

    /**
     * Pure Rust reset implementation (independent of Python bindings).
     *
     * Returns the initial observation vector.
     */
    #[instrument(skip(self))]
    pub fn reset_rs(&mut self) -> Vec<f64> {
        debug!("Resetting environment (Rust)");
        self.current_step = self.lookback;
        self.position = 0.0;
        self.cash = self.initial_capital;
        self.returns.clear();
        self.prev_portfolio_value = self.initial_capital;
        self.orderbook.clear();
        self.risk_manager.reset(self.initial_capital);
        self.total_steps = 0;

        self.generate_observation_data()
    }

    /** Get current portfolio value */
    pub fn portfolio_value(&self) -> f64 {
        let price = self.current_price().unwrap_or(0.0);
        self.cash + self.position * price
    }

    /** Get current position */
    pub fn current_position(&self) -> f64 {
        self.position
    }

    /** Get current cash */
    pub fn current_cash(&self) -> f64 {
        self.cash
    }

    /** Get total steps taken */
    pub fn get_total_steps(&self) -> u64 {
        self.total_steps
    }

    /** Get reference to orderbook */
    pub fn orderbook(&self) -> &OrderBook {
        &self.orderbook
    }

    pub fn risk_status(&self) -> RiskStatus {
        self.risk_manager.status().clone()
    }

    /** Pure Rust step function (no Python dependency) */
    #[instrument(skip(self), fields(step = self.total_steps))]
    pub fn step_rs(&mut self, action: i32) -> (Vec<f64>, f64, bool, bool, StepInfo) {
        let action_type = ActionType::from(action);

        // Execute action
        let trade_cost = self.execute_action(action_type).unwrap_or(0.0);

        // Advance time
        self.current_step += 1;
        self.total_steps += 1;

        // Calculate reward
        let portfolio_value = self.portfolio_value();
        let returns = safe_div(
            portfolio_value - self.prev_portfolio_value,
            self.prev_portfolio_value,
            0.0,
        );
        self.returns.push(returns);
        self.prev_portfolio_value = portfolio_value;

        // Update risk manager
        self.risk_manager.update(portfolio_value);

        // Risk-adjusted reward (Sharpe-like)
        let reward = self.calculate_reward(returns, trade_cost);

        // Check termination
        let terminated =
            portfolio_value <= 0.0 || self.current_step >= self.prices.len().saturating_sub(1);
        let truncated = self.current_step.saturating_sub(self.lookback) >= self.max_steps;

        // Generate observation data
        let obs_data = self.generate_observation_data();

        // Collect info data
        let step_info = StepInfo {
            portfolio_value,
            position: self.position,
            cash: self.cash,
            sharpe_ratio: self.calculate_sharpe(30),
            total_steps: self.total_steps,
        };

        // Log simulation state (thread-safe)
        self.logger
            .log_step(self.total_steps, &self.orderbook, portfolio_value);

        (obs_data, reward, terminated, truncated, step_info)
    }

    /** Internal method to generate observation data (pure Rust, no GIL) */
    fn generate_observation_data(&mut self) -> Vec<f64> {
        self.obs_buffer.reset();

        for i in 0..self.lookback {
            let idx = self
                .current_step
                .saturating_sub(self.lookback)
                .saturating_add(i);
            if idx < self.prices.len() {
                let price = self.prices[idx];
                let prev_price = if idx > 0 { self.prices[idx - 1] } else { price };
                let returns = if prev_price > 0.0 {
                    (price - prev_price) / prev_price
                } else {
                    0.0
                };

                let row_data = [
                    SafeFloat::new(safe_div(price, self.prices[0], 1.0)).unwrap(), // Normalized price
                    SafeFloat::new(returns).unwrap(),                              // Returns
                    0.0,                                                 // Volume (placeholder)
                    SafeFloat::new(self.orderbook.imbalance()).unwrap(), // Order book imbalance
                    SafeFloat::new(safe_div(self.position, self.initial_capital, 0.0)).unwrap(), // Normalized position
                    SafeFloat::new(safe_div(self.cash, self.initial_capital, 1.0)).unwrap(), // Normalized cash
                ];
                self.obs_buffer.update(i, &row_data);
            }
        }
        self.obs_buffer.to_vec()
    }

    /** Get current price */
    fn current_price(&self) -> Option<f64> {
        self.prices.get(self.current_step).copied()
    }

    /** Execute a trading action */
    fn execute_action(&mut self, action: ActionType) -> ArenaResult<f64> {
        let price = match self.current_price() {
            Some(p) => p,
            None => return Ok(0.0),
        };

        let trade_size = self.initial_capital * 0.1; // Fixed position sizing
        let mut cost = 0.0;

        match action {
            ActionType::Hold => {}
            ActionType::Buy => {
                let shares = safe_div(trade_size, price, 0.0);
                let tx_cost = trade_size * self.transaction_cost;

                if self.cash >= trade_size + tx_cost {
                    self.cash -= trade_size + tx_cost;
                    self.position += shares;
                    cost = tx_cost;

                    // Update orderbook with our trade
                    self.orderbook.submit_market_order(shares, Side::Bid)?;
                }
            }
            ActionType::Sell => {
                if self.position > 0.0 {
                    let shares_to_sell = safe_div(trade_size, price, 0.0).min(self.position);
                    let proceeds = shares_to_sell * price;
                    let tx_cost = proceeds * self.transaction_cost;

                    self.position -= shares_to_sell;
                    self.cash += proceeds - tx_cost;
                    cost = tx_cost;

                    self.orderbook
                        .submit_market_order(shares_to_sell, Side::Ask)?;
                }
            }
        }

        Ok(cost)
    }

    /** Calculate risk-adjusted reward */
    fn calculate_reward(&self, returns: f64, trade_cost: f64) -> f64 {
        // Simple reward: returns minus transaction costs
        let base_reward = returns * 100.0; // Scale up small returns

        // Penalty for holding costs (encourage action when appropriate)
        let cost_penalty = trade_cost / self.initial_capital * 100.0;

        // Drawdown penalty
        let max_value: f64 = self
            .returns
            .iter()
            .scan(self.initial_capital, |acc, &r| {
                *acc *= 1.0 + r;
                Some(*acc)
            })
            .fold(self.initial_capital, f64::max);

        let current_value = self.portfolio_value();
        let drawdown = if max_value > 0.0 {
            (max_value - current_value) / max_value
        } else {
            0.0
        };
        let drawdown_penalty = drawdown * 0.1;

        base_reward - cost_penalty - drawdown_penalty
    }

    /** Calculate rolling Sharpe ratio */
    fn calculate_sharpe(&self, window: usize) -> f64 {
        if self.returns.len() < 2 {
            return 0.0;
        }

        let recent: Vec<_> = self.returns.iter().rev().take(window).copied().collect();

        if recent.len() < 2 {
            return 0.0;
        }

        let mean: f64 = recent.iter().sum::<f64>() / recent.len() as f64;
        let variance: f64 =
            recent.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (recent.len() - 1) as f64;
        let std = variance.sqrt();

        if std > 0.0 {
            mean / std * (252.0_f64).sqrt() // Annualized
        } else {
            0.0
        }
    }

    /**
     * Set the random seed for reproducibility.
     *
     * # Arguments
     *
     * * `seed` - Random seed value.
     */
    pub fn set_seed(&mut self, seed: u64) {
        self.rng = StdRng::seed_from_u64(seed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_env_creation() {
        let env = TradingEnv::new(10000.0, 0.001, 30, 1000, true, Some(42));
        assert_eq!(env.portfolio_value(), 10000.0);
        assert_eq!(env.observation_shape(), (30, 6));
    }

    #[test]
    fn test_action_conversion() {
        assert!(matches!(ActionType::from(0), ActionType::Hold));
        assert!(matches!(ActionType::from(1), ActionType::Buy));
        assert!(matches!(ActionType::from(2), ActionType::Sell));
        assert!(matches!(ActionType::from(99), ActionType::Hold)); // Invalid defaults to Hold
    }
}
