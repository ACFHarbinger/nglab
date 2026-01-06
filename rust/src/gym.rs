//! Gymnasium-compatible trading environment
//!
//! Implements the standard RL interface (reset, step) with:
//! - Discrete and continuous action spaces
//! - Normalized observations via zero-copy numpy arrays
//! - Risk-adjusted reward functions

use crate::orderbook::{OrderBook, Side};
use numpy::{PyArray2, ToPyArray};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Action space types
#[derive(Debug, Clone, Copy)]
pub enum ActionType {
    /// Hold current position
    Hold,
    /// Buy (long)
    Buy,
    /// Sell (short)
    Sell,
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

/// Step result returned to Python
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StepResult {
    pub observation: Vec<f64>,
    pub reward: f64,
    pub terminated: bool,
    pub truncated: bool,
    pub info: StepInfo,
}

/// Additional info for each step
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct StepInfo {
    pub portfolio_value: f64,
    pub position: f64,
    pub cash: f64,
    pub sharpe_ratio: f64,
    pub total_steps: u64,
}

/// Observation buffer for zero-copy access
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

    pub fn update(&mut self, row: usize, values: &[f64]) {
        let start = row * self.shape.1;
        let end = start + values.len().min(self.shape.1);
        self.data[start..end].copy_from_slice(&values[..end - start]);
    }
}

/// Trading environment with Gymnasium interface
#[pyclass]
pub struct TradingEnv {
    /// Order book for simulation
    orderbook: OrderBook,
    /// Price history for backtesting
    prices: Vec<f64>,
    /// Current step index
    current_step: usize,
    /// Current position (positive = long, negative = short)
    position: f64,
    /// Current cash balance
    cash: f64,
    /// Initial capital
    initial_capital: f64,
    /// Transaction cost rate
    transaction_cost: f64,
    /// Lookback window for observations
    lookback: usize,
    /// Number of features per timestep
    num_features: usize,
    /// Returns history for Sharpe calculation
    returns: Vec<f64>,
    /// Previous portfolio value for return calculation
    prev_portfolio_value: f64,
    /// Maximum steps before truncation
    max_steps: usize,
    /// Total steps taken
    total_steps: u64,
    /// Rerun logger for visualization
    logger: crate::visualizer::RerunLogger,
}

#[pymethods]
impl TradingEnv {
    #[new]
    #[pyo3(signature = (initial_capital=10000.0, transaction_cost=0.001, lookback=30, max_steps=1000))]
    pub fn new(
        initial_capital: f64,
        transaction_cost: f64,
        lookback: usize,
        max_steps: usize,
    ) -> Self {
        TradingEnv {
            orderbook: OrderBook::new(),
            prices: Vec::new(),
            current_step: 0,
            position: 0.0,
            cash: initial_capital,
            initial_capital,
            transaction_cost,
            lookback,
            num_features: 6, // price, return, volume, imbalance, position, cash
            returns: Vec::new(),
            prev_portfolio_value: initial_capital,
            max_steps,
            total_steps: 0,
            logger: crate::visualizer::RerunLogger::new(true), // Enable by default for now
        }
    }

    /// Load price data for backtesting
    pub fn load_prices(&mut self, prices: Vec<f64>) {
        self.prices = prices;
    }

    /// Get observation space shape
    pub fn observation_shape(&self) -> (usize, usize) {
        (self.lookback, self.num_features)
    }

    /// Get action space size (discrete: 0=hold, 1=buy, 2=sell)
    pub fn action_space_size(&self) -> usize {
        3
    }

    /// Reset the environment
    pub fn reset<'py>(&mut self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.current_step = self.lookback;
        self.position = 0.0;
        self.cash = self.initial_capital;
        self.returns.clear();
        self.prev_portfolio_value = self.initial_capital;
        self.orderbook.clear();

        self.get_observation(py)
    }

    /// Take a step in the environment
    pub fn step<'py>(
        &mut self,
        py: Python<'py>,
        action: i32,
    ) -> (Bound<'py, PyArray2<f64>>, f64, bool, bool, PyObject) {
        let action_type = ActionType::from(action);
        let lookback = self.lookback;
        let num_features = self.num_features;

        // Run simulation logic without GIL
        let (reward, terminated, truncated, obs_data, step_info) = py.allow_threads(move || {
            // Execute action
            let trade_cost = self.execute_action(action_type);

            // Advance time
            self.current_step += 1;
            self.total_steps += 1;

            // Calculate reward
            let portfolio_value = self.portfolio_value();
            let returns = (portfolio_value - self.prev_portfolio_value) / self.prev_portfolio_value;
            self.returns.push(returns);
            self.prev_portfolio_value = portfolio_value;

            // Risk-adjusted reward (Sharpe-like)
            let reward = self.calculate_reward(returns, trade_cost);

            // Check termination
            let terminated = portfolio_value <= 0.0 || self.current_step >= self.prices.len() - 1;
            let truncated = self.current_step - self.lookback >= self.max_steps;

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

            (reward, terminated, truncated, obs_data, step_info)
        });

        // Build info dict (needs GIL)
        let dict = PyDict::new_bound(py);
        dict.set_item("portfolio_value", step_info.portfolio_value)
            .unwrap();
        dict.set_item("position", step_info.position).unwrap();
        dict.set_item("cash", step_info.cash).unwrap();
        dict.set_item("sharpe_ratio", step_info.sharpe_ratio)
            .unwrap();
        dict.set_item("total_steps", step_info.total_steps).unwrap();
        let info = dict.into();

        // Create numpy array (needs GIL)
        let obs_array = ndarray::Array2::from_shape_vec((lookback, num_features), obs_data)
            .expect("Invalid shape");
        let obs = obs_array.to_pyarray_bound(py);

        (obs, reward, terminated, truncated, info)
    }

    /// Get current observation as numpy array
    pub fn get_observation<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let obs_data = self.generate_observation_data();

        // Reshape to (lookback, features)
        let array = ndarray::Array2::from_shape_vec((self.lookback, self.num_features), obs_data)
            .expect("Invalid shape");

        array.to_pyarray_bound(py)
    }

    /// Get current portfolio value
    pub fn portfolio_value(&self) -> f64 {
        let price = self.current_price().unwrap_or(0.0);
        self.cash + self.position * price
    }

    /// Get current position
    pub fn current_position(&self) -> f64 {
        self.position
    }

    /// Get current cash
    pub fn current_cash(&self) -> f64 {
        self.cash
    }

    /// Get total steps taken
    pub fn get_total_steps(&self) -> u64 {
        self.total_steps
    }
}

impl TradingEnv {
    /// Get reference to orderbook
    pub fn orderbook(&self) -> &OrderBook {
        &self.orderbook
    }

    /// Pure Rust step function (no Python dependency)
    pub fn step_rs(&mut self, action: i32) -> (Vec<f64>, f64, bool, bool, StepInfo) {
        let action_type = ActionType::from(action);

        // Execute action
        let trade_cost = self.execute_action(action_type);

        // Advance time
        self.current_step += 1;
        self.total_steps += 1;

        // Calculate reward
        let portfolio_value = self.portfolio_value();
        let returns = (portfolio_value - self.prev_portfolio_value) / self.prev_portfolio_value;
        self.returns.push(returns);
        self.prev_portfolio_value = portfolio_value;

        // Risk-adjusted reward (Sharpe-like)
        let reward = self.calculate_reward(returns, trade_cost);

        // Check termination
        let terminated =
            portfolio_value <= 0.0 || self.current_step >= self.prices.len().saturating_sub(1);
        let truncated = self.current_step - self.lookback >= self.max_steps;

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

    /// Internal method to generate observation data (pure Rust, no GIL)
    fn generate_observation_data(&self) -> Vec<f64> {
        let mut obs = vec![0.0f64; self.lookback * self.num_features];

        for i in 0..self.lookback {
            let idx = self.current_step - self.lookback + i;
            if idx < self.prices.len() {
                let price = self.prices[idx];
                let prev_price = if idx > 0 { self.prices[idx - 1] } else { price };
                let returns = if prev_price > 0.0 {
                    (price - prev_price) / prev_price
                } else {
                    0.0
                };

                let row_start = i * self.num_features;
                obs[row_start] = price / self.prices[0]; // Normalized price
                obs[row_start + 1] = returns; // Returns
                obs[row_start + 2] = 0.0; // Volume (placeholder)
                obs[row_start + 3] = self.orderbook.imbalance(); // Order book imbalance
                obs[row_start + 4] = self.position / self.initial_capital; // Normalized position
                obs[row_start + 5] = self.cash / self.initial_capital; // Normalized cash
            }
        }
        obs
    }
}

impl TradingEnv {
    /// Get current price
    fn current_price(&self) -> Option<f64> {
        self.prices.get(self.current_step).copied()
    }

    /// Execute a trading action
    fn execute_action(&mut self, action: ActionType) -> f64 {
        let price = match self.current_price() {
            Some(p) => p,
            None => return 0.0,
        };

        let trade_size = self.initial_capital * 0.1; // Fixed position sizing
        let mut cost = 0.0;

        match action {
            ActionType::Hold => {}
            ActionType::Buy => {
                let shares = trade_size / price;
                let tx_cost = trade_size * self.transaction_cost;

                if self.cash >= trade_size + tx_cost {
                    self.cash -= trade_size + tx_cost;
                    self.position += shares;
                    cost = tx_cost;

                    // Update orderbook with our trade
                    self.orderbook.submit_market_order(shares, Side::Bid);
                }
            }
            ActionType::Sell => {
                if self.position > 0.0 {
                    let shares_to_sell = (trade_size / price).min(self.position);
                    let proceeds = shares_to_sell * price;
                    let tx_cost = proceeds * self.transaction_cost;

                    self.position -= shares_to_sell;
                    self.cash += proceeds - tx_cost;
                    cost = tx_cost;

                    self.orderbook
                        .submit_market_order(shares_to_sell, Side::Ask);
                }
            }
        }

        cost
    }

    /// Calculate risk-adjusted reward
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

    /// Calculate rolling Sharpe ratio
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

    /// Build info dictionary for Python
    #[allow(dead_code)]
    fn build_info(&self, py: Python<'_>) -> PyObject {
        let dict = PyDict::new_bound(py);
        dict.set_item("portfolio_value", self.portfolio_value())
            .unwrap();
        dict.set_item("position", self.position).unwrap();
        dict.set_item("cash", self.cash).unwrap();
        dict.set_item("sharpe_ratio", self.calculate_sharpe(30))
            .unwrap();
        dict.set_item("total_steps", self.total_steps).unwrap();
        dict.into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_env_creation() {
        let env = TradingEnv::new(10000.0, 0.001, 30, 1000);
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
