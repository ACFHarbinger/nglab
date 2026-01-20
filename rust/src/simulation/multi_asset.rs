#[cfg(feature = "python")]
use crate::errors::ArenaError;
use crate::simulation::gym::ActionType;
use crate::simulation::orderbook::{OrderBook, Side, Trade};
use crate::simulation::risk::{RiskManager, RiskStatus};
#[cfg(feature = "python")]
use numpy::{PyArray2, ToPyArray};
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyDict;
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[cfg_attr(feature = "python", pyclass)]
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum AlgoType {
    TWAP,
    VWAP,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgoOrder {
    pub asset: String,
    pub side: Side,
    pub total_quantity: f64,
    pub remaining_quantity: f64,
    pub start_step: usize,
    pub end_step: usize,
    pub algo_type: AlgoType,
}

// We reuse ActionType, StepInfo, ObservationBuffer from gym (or redefine if needed)
// ActionType is generic enough.
// StepInfo is generic enough but might need per-asset breakdown.

#[cfg_attr(feature = "python", pyclass)]
pub struct MultiAssetEnv {
    assets: Vec<String>,
    orderbooks: HashMap<String, OrderBook>,
    prices: HashMap<String, Vec<f64>>,
    current_step: usize,
    positions: HashMap<String, f64>,
    cash: f64,
    initial_capital: f64,
    transaction_cost: f64,
    lookback: usize,
    max_steps: usize,
    total_steps: u64,
    // num_features per asset
    features_per_asset: usize,
    rng: StdRng,
    algo_orders: Vec<AlgoOrder>,
    risk_manager: RiskManager,
}

impl MultiAssetEnv {
    pub fn new(
        assets: Vec<String>,
        initial_capital: f64,
        transaction_cost: f64,
        lookback: usize,
        max_steps: usize,
        seed: Option<u64>,
    ) -> Self {
        let mut orderbooks = HashMap::new();
        let mut positions = HashMap::new();
        let mut prices = HashMap::new();

        for asset in &assets {
            orderbooks.insert(asset.clone(), OrderBook::new());
            positions.insert(asset.clone(), 0.0);
            prices.insert(asset.clone(), Vec::new());
        }

        let rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => {
                use rand::Rng;
                let random_seed = rand::rng().random::<u64>();
                StdRng::seed_from_u64(random_seed)
            }
        };

        MultiAssetEnv {
            assets,
            orderbooks,
            prices,
            current_step: 0,
            positions,
            cash: initial_capital,
            initial_capital,
            transaction_cost,
            lookback,
            max_steps,
            total_steps: 0,
            features_per_asset: 6, // price, return, vol, imb, pos, padding
            rng,
            algo_orders: Vec::new(),
            risk_manager: RiskManager::with_defaults(initial_capital),
        }
    }

    pub fn load_prices(&mut self, asset: String, prices: Vec<f64>) {
        self.prices.insert(asset, prices);
    }

    pub fn portfolio_value(&self) -> f64 {
        let mut value = self.cash;
        for (asset, pos) in &self.positions {
            if let Some(price_vec) = self.prices.get(asset) {
                if let Some(price) = price_vec.get(self.current_step) {
                    value += pos * price;
                }
            }
        }
        value
    }

    pub fn reset_native(&mut self, seed: Option<u64>) -> (Vec<f64>, HashMap<String, f64>) {
        if let Some(s) = seed {
            self.rng = StdRng::seed_from_u64(s);
        }

        self.current_step = self.lookback;
        self.total_steps = 0;
        self.cash = self.initial_capital;
        for pos in self.positions.values_mut() {
            *pos = 0.0;
        }
        for ob in self.orderbooks.values_mut() {
            ob.clear();
        }
        self.risk_manager.reset(self.initial_capital);

        let obs_data = self.generate_observation_data();
        let mut info = HashMap::new();
        info.insert("portfolio_value".to_string(), self.portfolio_value());
        info.insert("cash".to_string(), self.cash);

        (obs_data, info)
    }

    pub fn step_native(
        &mut self,
        actions: Vec<i32>,
    ) -> (Vec<f64>, f64, bool, bool, HashMap<String, f64>) {
        let prev_val = self.portfolio_value();

        // Execute actions per asset
        for (i, &action_idx) in actions.iter().enumerate() {
            if i >= self.assets.len() {
                break;
            }
            let action = ActionType::from(action_idx);
            let asset_name = self.assets[i].clone();

            // Seed orderbook from tape price if empty
            if let Some(price_vec) = self.prices.get(&asset_name) {
                if let Some(&price) = price_vec.get(self.current_step) {
                    self.seed_orderbook(&asset_name, price);

                    // Trigger advanced orders if any
                    if let Some(ob) = self.orderbooks.get_mut(&asset_name) {
                        ob.check_triggers(price);
                    }
                }
            }

            self.execute_asset_action(i, action);
        }

        // Process Algo Orders
        self.process_algo_orders();

        self.current_step += 1;
        self.total_steps += 1;

        let new_val = self.portfolio_value();
        let returns = if prev_val > 0.0 {
            (new_val - prev_val) / prev_val
        } else {
            0.0
        };
        let reward = returns * 100.0;

        let terminated = new_val <= 0.0;
        let truncated = self.total_steps as usize >= self.max_steps;

        // Update risk manager
        self.risk_manager.update(new_val);

        let obs_data = self.generate_observation_data();

        let mut info = HashMap::new();
        info.insert("portfolio_value".to_string(), new_val);
        info.insert("cash".to_string(), self.cash);

        let risk_status = self.risk_manager.status();
        info.insert("risk_score".to_string(), risk_status.risk_score as f64);
        info.insert("current_drawdown".to_string(), risk_status.current_drawdown);
        info.insert("current_var".to_string(), risk_status.current_var);

        (obs_data, reward, terminated, truncated, info)
    }

    pub fn risk_status(&self) -> RiskStatus {
        self.risk_manager.status().clone()
    }
}

// Python bindings
#[cfg(feature = "python")]
#[pymethods]
impl MultiAssetEnv {
    #[new]
    #[pyo3(signature = (assets, initial_capital=10000.0, transaction_cost=0.001, lookback=30, max_steps=1000, seed=None))]
    pub fn new_py(
        assets: Vec<String>,
        initial_capital: f64,
        transaction_cost: f64,
        lookback: usize,
        max_steps: usize,
        seed: Option<u64>,
    ) -> Self {
        Self::new(
            assets,
            initial_capital,
            transaction_cost,
            lookback,
            max_steps,
            seed,
        )
    }

    #[pyo3(name = "load_prices")]
    pub fn load_prices_py(&mut self, asset: String, prices: Vec<f64>) {
        self.load_prices(asset, prices);
    }

    #[pyo3(signature = (seed=None))]
    pub fn reset<'py>(
        &mut self,
        py: Python<'py>,
        seed: Option<u64>,
    ) -> PyResult<(Bound<'py, PyArray2<f64>>, Py<PyAny>)> {
        let (obs_data, info_map) = self.reset_native(seed);
        let total_features = self.assets.len() * self.features_per_asset;

        let obs_array = ndarray::Array2::from_shape_vec((self.lookback, total_features), obs_data)
            .map_err(|e| ArenaError::DataLoading(format!("Invalid shape: {}", e)))?;

        let info = PyDict::new(py);
        for (k, v) in info_map {
            info.set_item(k, v)?;
        }
        Ok((obs_array.to_pyarray(py), info.into()))
    }

    // Actions: List of integers (one per asset)
    #[allow(clippy::type_complexity)]
    pub fn step<'py>(
        &mut self,
        py: Python<'py>,
        actions: Vec<i32>,
    ) -> PyResult<(Bound<'py, PyArray2<f64>>, f64, bool, bool, Py<PyAny>)> {
        let (obs_data, reward, terminated, truncated, info_map) = self.step_native(actions);
        let total_features = self.assets.len() * self.features_per_asset;

        let obs_array = ndarray::Array2::from_shape_vec((self.lookback, total_features), obs_data)
            .map_err(|e| ArenaError::DataLoading(format!("Invalid shape: {}", e)))?;

        let info = PyDict::new(py);
        for (k, v) in info_map {
            info.set_item(k, v)?;
        }

        Ok((
            obs_array.to_pyarray(py),
            reward,
            terminated,
            truncated,
            info.into(),
        ))
    }
}

impl MultiAssetEnv {
    fn seed_orderbook(&mut self, asset: &str, price: f64) {
        if let Some(ob) = self.orderbooks.get_mut(asset) {
            if ob.best_bid().is_none() {
                // Add synthetic liquidity around the tape price (0.1% spread)
                let spread = price * 0.001;
                ob.submit_limit_order(price - spread / 2.0, 1000.0, Side::Bid);
                ob.submit_limit_order(price + spread / 2.0, 1000.0, Side::Ask);
            }
        }
    }

    fn execute_asset_action(&mut self, asset_idx: usize, action: ActionType) {
        let asset_name = self.assets[asset_idx].clone();
        let multiplier = self.risk_manager.status().position_multiplier;
        let trade_size_usd = self.initial_capital * 0.05 * multiplier;

        if multiplier <= 0.0 && action != ActionType::Hold {
            return;
        }

        match action {
            ActionType::Hold => {}
            ActionType::Buy => {
                if let Some(ob) = self.orderbooks.get_mut(&asset_name) {
                    // Estimate shares from tape price
                    let price = self
                        .prices
                        .get(&asset_name)
                        .and_then(|v| v.get(self.current_step))
                        .cloned()
                        .unwrap_or(1.0);

                    // Add some stochastic slippage (0-0.1%)
                    let slippage = self.rng.random_range(0.0..0.001);
                    let effective_price = price * (1.0 + slippage);

                    let target_shares = trade_size_usd / effective_price;

                    let (_, trades) = ob.submit_market_order(target_shares, Side::Bid);
                    self.apply_trades(&asset_name, trades);
                }
            }
            ActionType::Sell => {
                let pos = *self.positions.get(&asset_name).unwrap();
                if pos > 0.0 {
                    if let Some(ob) = self.orderbooks.get_mut(&asset_name) {
                        let price = self
                            .prices
                            .get(&asset_name)
                            .and_then(|v| v.get(self.current_step))
                            .cloned()
                            .unwrap_or(1.0);

                        // Add some stochastic slippage (0-0.1%)
                        let slippage = self.rng.random_range(0.0..0.001);
                        let effective_price = price * (1.0 - slippage);
                        let target_shares = (trade_size_usd / effective_price).min(pos);

                        let (_, trades) = ob.submit_market_order(target_shares, Side::Ask);
                        self.apply_trades(&asset_name, trades);
                    }
                }
            }
        }
    }

    fn apply_trades(&mut self, asset: &str, trades: Vec<Trade>) {
        for trade in trades {
            let cost = trade.price * trade.quantity;
            let fee = cost * self.transaction_cost;
            match trade.side {
                Side::Bid => {
                    // Taker was Bid, so we are Buying
                    if self.cash >= cost + fee {
                        self.cash -= cost + fee;
                        *self.positions.get_mut(asset).unwrap() += trade.quantity;
                    }
                }
                Side::Ask => {
                    // Taker was Ask, so we are Selling
                    self.cash += cost - fee;
                    *self.positions.get_mut(asset).unwrap() -= trade.quantity;
                }
            }
        }
    }

    fn process_algo_orders(&mut self) {
        let current_step = self.current_step;
        let mut all_completed = Vec::new();
        let mut trades_to_apply = Vec::new();

        // We iterate and slice
        for (i, algo) in self.algo_orders.iter_mut().enumerate() {
            if current_step < algo.start_step {
                continue;
            }
            if current_step > algo.end_step || algo.remaining_quantity <= 1e-8 {
                all_completed.push(i);
                continue;
            }

            let remaining_steps = (algo.end_step - current_step + 1) as f64;
            let slice = (algo.remaining_quantity / remaining_steps)
                .max(0.0)
                .min(algo.remaining_quantity);

            if slice > 0.0 {
                let asset = algo.asset.clone();
                let side = algo.side;
                if let Some(ob) = self.orderbooks.get_mut(&asset) {
                    ob.set_timestamp(self.total_steps);
                    let (_, trades) = ob.submit_market_order(slice, side);
                    let filled: f64 = trades.iter().map(|t| t.quantity).sum();

                    if !trades.is_empty() {
                        trades_to_apply.push((asset, trades));
                    }
                    algo.remaining_quantity -= filled;
                }
            }

            if algo.remaining_quantity <= 1e-8 {
                all_completed.push(i);
            }
        }

        // Apply trades outside the loop to avoid borrow conflict
        for (asset, trades) in trades_to_apply {
            self.apply_trades(&asset, trades);
        }

        // Remove completed
        for &idx in all_completed.iter().rev() {
            self.algo_orders.remove(idx);
        }
    }

    #[cfg(feature = "python")]
    pub fn submit_algo_order_py(
        &mut self,
        asset: String,
        side_idx: i32,
        quantity: f64,
        duration: usize,
        algo_type: AlgoType,
    ) {
        let side = if side_idx == 0 { Side::Bid } else { Side::Ask };
        self.algo_orders.push(AlgoOrder {
            asset,
            side,
            total_quantity: quantity,
            remaining_quantity: quantity,
            start_step: self.current_step,
            end_step: self.current_step + duration,
            algo_type,
        });
    }

    fn generate_observation_data(&self) -> Vec<f64> {
        let total_features = self.assets.len() * self.features_per_asset;
        let mut data = vec![0.0; self.lookback * total_features];

        for t in 0..self.lookback {
            let step_idx = self
                .current_step
                .saturating_sub(self.lookback)
                .saturating_add(t);
            let row_start = t * total_features;

            for (i, asset) in self.assets.iter().enumerate() {
                let asset_offset = i * self.features_per_asset;
                let idx = row_start + asset_offset;

                let price_series = self.prices.get(asset).unwrap();
                let price = *price_series.get(step_idx).unwrap_or(&0.0);

                // 1. Price (raw for now, normalize in Python)
                data[idx] = price;

                // 2. Log Return
                if step_idx > 0 {
                    let prev_price = *price_series.get(step_idx - 1).unwrap_or(&price);
                    data[idx + 1] = (price / prev_price.max(1e-6)).ln();
                } else {
                    data[idx + 1] = 0.0;
                }

                // 3. Volatility (Simple 20-period standard deviation approx)
                if step_idx >= 20 {
                    let slice = &price_series[step_idx - 20..=step_idx];
                    let mean = slice.iter().sum::<f64>() / 21.0;
                    let variance = slice.iter().map(|&p| (p - mean).powi(2)).sum::<f64>() / 21.0;
                    data[idx + 2] = variance.sqrt();
                } else {
                    data[idx + 2] = 0.0;
                }

                // 4. Orderbook Imbalance
                if let Some(ob) = self.orderbooks.get(asset) {
                    data[idx + 3] = ob.imbalance();
                }

                // 5. Normalized Position
                // (Current shares * current price) / portfolio value
                let p_val = self.portfolio_value();
                if p_val > 0.0 {
                    let pos_shares = *self.positions.get(asset).unwrap();
                    data[idx + 4] = (pos_shares * price) / p_val;
                }

                // 6. Padding/Reserved
                data[idx + 5] = 0.0;
            }
        }
        data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multi_asset_env_native_loop() {
        let assets = vec!["BTC".to_string(), "ETH".to_string()];
        let mut env = MultiAssetEnv::new(assets, 10000.0, 0.001, 10, 100, Some(42));

        let btc_prices = vec![100.0; 200];
        let eth_prices = vec![2000.0; 200];

        env.load_prices("BTC".to_string(), btc_prices);
        env.load_prices("ETH".to_string(), eth_prices);

        // Reset
        let (obs, info) = env.reset_native(None);
        assert_eq!(obs.len(), 10 * 2 * 6); // lookback * assets * features
        assert_eq!(*info.get("portfolio_value").unwrap(), 10000.0);

        // Step: Buy BTC, Buy ETH
        let (_obs, reward, terminated, truncated, info) = env.step_native(vec![1, 1]); // Buy, Buy

        assert!(!terminated);
        assert!(!truncated);
        assert!(reward != 0.0 || *info.get("portfolio_value").unwrap() == 10000.0);
        assert!(*info.get("cash").unwrap() < 10000.0);
        assert!(env.positions.get("BTC").unwrap() > &0.0);
        assert!(env.positions.get("ETH").unwrap() > &0.0);

        // Check features
    }

    #[test]
    fn test_twap_execution() {
        let assets = vec!["BTC".to_string()];
        let mut env = MultiAssetEnv::new(assets, 10000.0, 0.001, 10, 100, Some(42));
        env.load_prices("BTC".to_string(), vec![100.0; 200]);
        env.reset_native(None);

        // Submit TWAP order: Buy 10 BTC over 5 steps
        let start = env.current_step;
        env.algo_orders.push(AlgoOrder {
            asset: "BTC".to_string(),
            side: Side::Bid,
            total_quantity: 10.0,
            remaining_quantity: 10.0,
            start_step: start,
            end_step: start + 4,
            algo_type: AlgoType::TWAP,
        });

        // 5 steps should execute 2 BTC each (ideally)
        for _ in 0..5 {
            env.step_native(vec![0]); // Hold, but algo should execute
        }

        assert_eq!(*env.positions.get("BTC").unwrap(), 10.0);
        assert_eq!(env.algo_orders.len(), 0);
    }

    #[test]
    fn test_risk_integration() {
        let mut env = MultiAssetEnv::new(vec!["BTC".to_string()], 100_000.0, 0.0, 1, 100, None);
        env.load_prices("BTC".to_string(), vec![100.0, 100.0, 1.0, 1.0]);
        env.reset_native(None); // current_step = 1

        // Step 1: Seed book and Buy BTC to get exposure (at price 100.0)
        env.seed_orderbook("BTC", 100.0);
        for _ in 0..10 {
            env.execute_asset_action(0, ActionType::Buy); // 10 * 5% = 50% exposure
        }

        env.step_native(vec![0]); // Advances to step 2 (price 1.0)

        let status = env.risk_status();
        // 50% exposure * 99% drop ~= 49.5% drawdown
        assert!(status.current_drawdown >= 0.15);
        assert!(status.drawdown_breached);
        assert!(status.position_multiplier < 1.0);

        // Step 2: Try to Buy with reduced multiplier
        // Default buy is initial_capital * 0.05 = 5000 USD
        // multiplier should be 0.25 (see risk.rs) -> 1250 USD
        env.execute_asset_action(0, ActionType::Buy);

        let cash_before = env.cash;
        // Step 2: Try to Buy with reduced multiplier (should be 0)
        env.execute_asset_action(0, ActionType::Buy);

        // Multiplier should be 0, so cash should be unchanged
        assert_eq!(env.cash, cash_before);
    }
}
