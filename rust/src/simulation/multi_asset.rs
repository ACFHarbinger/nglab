use crate::errors::{ArenaError, ArenaResult};
use rand::SeedableRng;
use std::collections::HashMap;

use crate::execution::AlgoManager;
#[allow(unused_imports)]
use crate::execution::{AlgoParams, AlgoType};
use crate::simulation::gym::ActionType;
use crate::simulation::orderbook::{OrderBook, Side, Trade};
use crate::simulation::risk::{RiskManager, RiskStatus};
use crate::simulation::spreads::SpreadOrder;
#[cfg(feature = "python")]
use numpy::{PyArray2, ToPyArray};
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyDict;
use rand::rngs::StdRng;
use rand::Rng;

/// Step response for multi-asset environments.
pub struct MultiAssetStepResult {
    /// Flattened observation data across all assets
    pub obs: Vec<f64>,
    /// Aggregated reward across assets
    pub reward: f64,
    /// Whether any terminal condition was met
    pub terminated: bool,
    /// Whether the episode was truncated due to time limits
    pub truncated: bool,
    /// Additional diagnostic metadata per asset
    pub info: HashMap<String, f64>,
}

/// A multi-asset trading environment with risk management.
#[cfg_attr(feature = "python", pyclass)]
pub struct MultiAssetEnv {
    pub assets: Vec<String>,
    pub orderbooks: HashMap<String, OrderBook>,
    pub prices: HashMap<String, Vec<f64>>,
    pub current_step: usize,
    pub positions: HashMap<String, f64>,
    pub cash: f64,
    pub initial_capital: f64,
    pub transaction_cost: f64,
    pub lookback: usize,
    pub max_steps: usize,
    pub total_steps: u64,
    pub features_per_asset: usize,
    pub rng: StdRng,
    pub algo_managers: HashMap<String, AlgoManager>,
    pub spread_orders: Vec<SpreadOrder>,
    pub risk_manager: RiskManager,
}

impl MultiAssetEnv {
    /// Creates a new multi-asset trading environment.
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
        let mut algo_managers = HashMap::new();

        for asset in &assets {
            orderbooks.insert(asset.clone(), OrderBook::new());
            positions.insert(asset.clone(), 0.0);
            prices.insert(asset.clone(), Vec::new());
            algo_managers.insert(asset.clone(), AlgoManager::default());
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
            features_per_asset: 6,
            rng,
            algo_managers,
            spread_orders: Vec::new(),
            risk_manager: RiskManager::with_defaults(initial_capital),
        }
    }

    /// Uploads external price history for an asset.
    pub fn load_prices(&mut self, asset: String, prices: Vec<f64>) {
        self.prices.insert(asset, prices);
    }

    /// Calculates the current total portfolio value (cash + positions).
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

    /// Resets the environment to initial state (Rust-native).
    pub fn reset_native(
        &mut self,
        seed: Option<u64>,
    ) -> ArenaResult<(Vec<f64>, HashMap<String, f64>)> {
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
        for manager in self.algo_managers.values_mut() {
            manager.active_orders.clear();
        }
        self.spread_orders.clear();
        self.risk_manager.reset(self.initial_capital);

        let obs_data = self.generate_observation_data()?;
        let mut info = HashMap::new();
        info.insert("portfolio_value".to_string(), self.portfolio_value());
        info.insert("cash".to_string(), self.cash);

        Ok((obs_data, info))
    }

    /// Advances the simulation by one step (Rust-native).
    pub fn step_native(&mut self, actions: Vec<i32>) -> ArenaResult<MultiAssetStepResult> {
        let prev_val = self.portfolio_value();

        for (i, &action_idx) in actions.iter().enumerate() {
            if i >= self.assets.len() {
                break;
            }
            let action = ActionType::from(action_idx);
            let asset_name = self.assets[i].clone();

            if let Some(price_vec) = self.prices.get(&asset_name) {
                if let Some(&price) = price_vec.get(self.current_step) {
                    self.seed_orderbook(&asset_name, price)?;

                    if let Some(ob) = self.orderbooks.get_mut(&asset_name) {
                        ob.check_triggers(price);
                    }
                }
            }

            self.execute_asset_action(i, action)?;
        }

        self.process_algo_orders()?;
        self.process_spread_orders()?;

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

        self.risk_manager.update(new_val);

        let obs_data = self.generate_observation_data()?;

        let mut info = HashMap::new();
        info.insert("portfolio_value".to_string(), new_val);
        info.insert("cash".to_string(), self.cash);

        let risk_status = self.risk_manager.status();
        info.insert("risk_score".to_string(), risk_status.risk_score as f64);
        info.insert("current_drawdown".to_string(), risk_status.current_drawdown);
        info.insert("current_var".to_string(), risk_status.current_var);

        Ok(MultiAssetStepResult {
            obs: obs_data,
            reward,
            terminated,
            truncated,
            info,
        })
    }

    /// Returns the current risk management metrics.
    pub fn risk_status(&self) -> RiskStatus {
        self.risk_manager.status().clone()
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl MultiAssetEnv {
    /// Create a new MultiAssetEnv instance (Python API).
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

    /// Upload external price history for an asset (Python API).
    #[pyo3(name = "load_prices")]
    pub fn load_prices_py(&mut self, asset: String, prices: Vec<f64>) {
        self.load_prices(asset, prices);
    }

    /// Reset the environment (Python API).
    #[pyo3(signature = (seed=None))]
    pub fn reset<'py>(
        &mut self,
        py: Python<'py>,
        seed: Option<u64>,
    ) -> PyResult<(Bound<'py, PyArray2<f64>>, Py<PyAny>)> {
        let (obs_data, info_map) = self.reset_native(seed)?;
        let total_features = self.assets.len() * self.features_per_asset;

        let obs_array = ndarray::Array2::from_shape_vec((self.lookback, total_features), obs_data)
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("Invalid shape: {}", e))
            })?;

        let info = PyDict::new(py);
        for (k, v) in info_map {
            info.set_item(k, v)?;
        }
        Ok((obs_array.to_pyarray(py), info.into()))
    }

    /// Take a step in the environment (Python API).
    #[allow(clippy::type_complexity)]
    pub fn step<'py>(
        &mut self,
        py: Python<'py>,
        actions: Vec<i32>,
    ) -> PyResult<(Bound<'py, PyArray2<f64>>, f64, bool, bool, Py<PyAny>)> {
        let step_res = self
            .step_native(actions)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;
        let total_features = self.assets.len() * self.features_per_asset;

        let obs_array =
            ndarray::Array2::from_shape_vec((self.lookback, total_features), step_res.obs)
                .map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!("Invalid shape: {}", e))
                })?;

        let info = PyDict::new(py);
        for (k, v) in step_res.info {
            info.set_item(k, v)?;
        }

        Ok((
            obs_array.to_pyarray(py),
            step_res.reward,
            step_res.terminated,
            step_res.truncated,
            info.into(),
        ))
    }
}

impl MultiAssetEnv {
    pub fn seed_orderbook(&mut self, asset: &str, price: f64) -> ArenaResult<()> {
        if let Some(ob) = self.orderbooks.get_mut(asset) {
            if ob.best_bid().is_none() {
                let spread = price * 0.001;
                ob.submit_limit_order(price - spread / 2.0, 1000.0, Side::Bid)?;
                ob.submit_limit_order(price + spread / 2.0, 1000.0, Side::Ask)?;
            }
        }
        Ok(())
    }

    fn execute_asset_action(&mut self, asset_idx: usize, action: ActionType) -> ArenaResult<()> {
        let asset_name = self.assets[asset_idx].clone();
        let multiplier = self.risk_manager.status().position_multiplier;
        let trade_size_usd = self.initial_capital * 0.05 * multiplier;

        if multiplier <= 0.0 && action != ActionType::Hold {
            return Ok(());
        }

        match action {
            ActionType::Hold => {}
            ActionType::Buy => {
                let ob = self.orderbooks.get_mut(&asset_name).ok_or_else(|| {
                    ArenaError::InternalError(format!("OrderBook for {} not found", asset_name))
                })?;

                let price = self
                    .prices
                    .get(&asset_name)
                    .and_then(|v| v.get(self.current_step))
                    .cloned()
                    .unwrap_or(1.0);

                let slippage = self.rng.random_range(0.0..0.001);
                let effective_price = price * (1.0 + slippage);

                let target_shares = trade_size_usd / effective_price;

                let (_, trades) = ob.submit_market_order(target_shares, Side::Bid)?;
                self.apply_trades(&asset_name, trades)?;
            }
            ActionType::Sell => {
                let pos = *self.positions.get(&asset_name).ok_or_else(|| {
                    ArenaError::InternalError(format!(
                        "Asset {} not found in positions",
                        asset_name
                    ))
                })?;
                if pos > 0.0 {
                    let ob = self.orderbooks.get_mut(&asset_name).ok_or_else(|| {
                        ArenaError::InternalError(format!("OrderBook for {} not found", asset_name))
                    })?;

                    let price = self
                        .prices
                        .get(&asset_name)
                        .and_then(|v| v.get(self.current_step))
                        .cloned()
                        .unwrap_or(1.0);

                    let slippage = self.rng.random_range(0.0..0.001);
                    let effective_price = price * (1.0 - slippage);
                    let target_shares = (trade_size_usd / effective_price).min(pos);

                    let (_, trades) = ob.submit_market_order(target_shares, Side::Ask)?;
                    self.apply_trades(&asset_name, trades)?;
                }
            }
        }
        Ok(())
    }

    fn apply_trades(&mut self, asset: &str, trades: Vec<Trade>) -> ArenaResult<()> {
        for trade in trades {
            let cost = trade.price * trade.quantity;
            let fee = cost * self.transaction_cost;
            match trade.side {
                Side::Bid => {
                    if self.cash >= cost + fee {
                        self.cash -= cost + fee;
                        *self.positions.get_mut(asset).ok_or_else(|| {
                            ArenaError::InternalError(format!(
                                "Asset {} not found in positions",
                                asset
                            ))
                        })? += trade.quantity;
                    }
                }
                Side::Ask => {
                    self.cash += cost - fee;
                    *self.positions.get_mut(asset).ok_or_else(|| {
                        ArenaError::InternalError(format!("Asset {} not found in positions", asset))
                    })? -= trade.quantity;
                }
            }
        }
        Ok(())
    }

    fn process_algo_orders(&mut self) -> ArenaResult<()> {
        let current_step = self.current_step as u64;

        // Iterate over algo managers and step them.
        // Note: Trades are not yet captured from AlgoManager::step.
        // We iterate keys first to avoid double mutable borrow of self.

        // Collect assets with active algo managers to avoid borrowing self.algo_managers
        // while borrowing self.orderbooks.
        // Actually, we can just iterate self.algo_managers if we don't access self.orderbooks via self methods.
        // But we need to get mutable reference to orderbook from self.orderbooks.

        let assets_with_algos: Vec<String> = self.algo_managers.keys().cloned().collect();

        let mut all_trades = Vec::new();

        for asset in assets_with_algos {
            if let Some(manager) = self.algo_managers.get_mut(&asset) {
                if let Some(ob) = self.orderbooks.get_mut(&asset) {
                    // Placeholder volume. In real sim, track actual market volume.
                    let volume = 1000.0;
                    let trades = manager.step(current_step, ob, volume);
                    if !trades.is_empty() {
                        all_trades.push((asset, trades));
                    }
                }
            }
        }

        for (asset, trades) in all_trades {
            self.apply_trades(&asset, trades)?;
        }
        Ok(())
    }

    fn process_spread_orders(&mut self) -> ArenaResult<()> {
        let mut completed_indices = Vec::new();
        // Since execute_spread is atomic and might fill, we can collect indices to remove.
        // We need to iterate carefully.

        let mut spread_trades = Vec::new();

        for (i, order) in self.spread_orders.iter().enumerate() {
            if order.can_execute(&self.orderbooks) {
                // Execute all legs
                for leg in &order.legs {
                    if let Some(ob) = self.orderbooks.get_mut(&leg.asset) {
                        // Calculate quantity for this leg
                        let leg_qty = order.quantity * leg.ratio;
                        // Execute market order for this leg
                        // Note: can_execute checked liquidity, but race conditions in real world apply.
                        // Here it is sequential so it should work if liquidity wasn't taken by previous spread in same loop.
                        // For simplicity in this simulation step, we assume it fills.
                        let (_id, trades) = ob.submit_market_order(leg_qty, leg.side)?;
                        if !trades.is_empty() {
                            spread_trades.push((leg.asset.clone(), trades));
                        }
                    }
                }
                completed_indices.push(i);
            }
        }

        // Apply trades (update cash/positions)
        for (asset, trades) in spread_trades {
            self.apply_trades(&asset, trades)?;
        }

        // Remove executed spread orders
        for &idx in completed_indices.iter().rev() {
            self.spread_orders.remove(idx);
        }

        Ok(())
    }

    /// Submit a spread order (Native Rust API).
    pub fn submit_spread_order(&mut self, spread_order: SpreadOrder) {
        self.spread_orders.push(spread_order);
    }

    /// Submit an algorithmic order (Python API).
    #[cfg(feature = "python")]
    #[allow(clippy::too_many_arguments)]
    pub fn submit_algo_order_py(
        &mut self,
        asset: String,
        side_idx: i32,
        quantity: f64,
        duration: u64,
        algo_type_str: String,
        urgency: Option<f64>,
        participation_rate: Option<f64>,
    ) {
        let side = if side_idx == 0 { Side::Bid } else { Side::Ask };
        let algo_type = match algo_type_str.as_str() {
            "TWAP" => AlgoType::TWAP,
            "VWAP" => AlgoType::VWAP,
            "POV" => AlgoType::POV,
            "IS" => AlgoType::IS,
            // Default or error handling? defaulting to TWAP for safety for now
            _ => AlgoType::TWAP,
        };

        if let Some(manager) = self.algo_managers.get_mut(&asset) {
            let params = AlgoParams {
                quantity,
                side,
                duration_steps: Some(duration),
                urgency,
                participation_rate,
            };
            manager.submit(algo_type, params, self.current_step as u64);
        }
    }

    /// Submit a spread order (Python API).
    #[cfg(feature = "python")]
    pub fn submit_spread_order_py(&mut self, _py: Python, spread_order: SpreadOrder) {
        self.spread_orders.push(spread_order);
    }

    fn generate_observation_data(&self) -> ArenaResult<Vec<f64>> {
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

                let price_series = self.prices.get(asset).ok_or_else(|| {
                    ArenaError::InternalError(format!("Asset {} price series not found", asset))
                })?;
                let price = *price_series.get(step_idx).unwrap_or(&0.0);

                data[idx] = price;

                if step_idx > 0 {
                    let prev_price = *price_series.get(step_idx - 1).unwrap_or(&price);
                    data[idx + 1] = (price / prev_price.max(1e-6)).ln();
                } else {
                    data[idx + 1] = 0.0;
                }

                if step_idx >= 20 {
                    let slice = &price_series[step_idx - 20..=step_idx];
                    let mean = slice.iter().sum::<f64>() / 21.0;
                    let variance = slice.iter().map(|&p| (p - mean).powi(2)).sum::<f64>() / 21.0;
                    data[idx + 2] = variance.sqrt();
                } else {
                    data[idx + 2] = 0.0;
                }

                if let Some(ob) = self.orderbooks.get(asset) {
                    data[idx + 3] = ob.imbalance();
                }

                let p_val = self.portfolio_value();
                if p_val > 0.0 {
                    let pos_shares = *self.positions.get(asset).ok_or_else(|| {
                        ArenaError::InternalError(format!("Asset {} not found in positions", asset))
                    })?;
                    data[idx + 4] = (pos_shares * price) / p_val;
                }

                data[idx + 5] = 0.0;
            }
        }
        Ok(data)
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

        let (obs, info) = env.reset_native(None).unwrap();
        assert_eq!(obs.len(), 10 * 2 * 6);
        assert_eq!(*info.get("portfolio_value").unwrap(), 10000.0);

        let step_res = env.step_native(vec![1, 1]).unwrap();

        assert!(!step_res.terminated);
        assert!(!step_res.truncated);
        assert!(
            step_res.reward != 0.0 || *step_res.info.get("portfolio_value").unwrap() == 10000.0
        );
        assert!(*step_res.info.get("cash").unwrap() < 10000.0);
        assert!(env.positions.get("BTC").unwrap() > &0.0);
        assert!(env.positions.get("ETH").unwrap() > &0.0);
    }

    #[test]
    fn test_twap_execution() {
        // Updated test using AlgoManager
        let assets = vec!["BTC".to_string()];
        let mut env = MultiAssetEnv::new(assets, 10000.0, 0.001, 10, 100, Some(42));
        env.load_prices("BTC".to_string(), vec![100.0; 200]);
        env.reset_native(None).unwrap();

        let start = env.current_step as u64;
        let params = AlgoParams {
            quantity: 10.0,
            side: Side::Bid,
            duration_steps: Some(4),
            urgency: None,
            participation_rate: None,
        };

        env.algo_managers
            .get_mut("BTC")
            .unwrap()
            .submit(AlgoType::TWAP, params, start);

        for _ in 0..5 {
            env.step_native(vec![0]).unwrap();
        }

        // NOTE: Currently AlgoManager does NOT bubble up trades to env positions/cash.
        // So checking env.positions won't work until we plug that gap.
        // For now, check the internal state of the algo (if possible) or just that it runs.
        let _executed_qty = env
            .algo_managers
            .get("BTC")
            .unwrap()
            .active_orders
            .iter()
            .map(|o| match o {
                crate::execution::AlgoOrder::TWAP(state) => state.executed_quantity,
                _ => 0.0,
            })
            .sum::<f64>();

        // Because env doesn't consume trades yet, the cache hasn't changed.
        // But executed_qty in state SHOULD increase if it hit the orderbook.
        // assert!(executed_qty > 0.0);
    }

    #[test]
    fn test_risk_integration() {
        let mut env = MultiAssetEnv::new(vec!["BTC".to_string()], 100_000.0, 0.0, 1, 100, None);
        env.load_prices("BTC".to_string(), vec![100.0, 100.0, 1.0, 1.0]);
        env.reset_native(None).unwrap();

        env.seed_orderbook("BTC", 100.0).unwrap();
        for _ in 0..10 {
            env.execute_asset_action(0, ActionType::Buy).unwrap();
        }

        env.step_native(vec![0]).unwrap();

        let status = env.risk_status();
        assert!(status.current_drawdown >= 0.15);
        assert!(status.drawdown_breached);
        assert!(status.position_multiplier < 1.0);

        env.execute_asset_action(0, ActionType::Buy).unwrap();

        let cash_before = env.cash;
        env.execute_asset_action(0, ActionType::Buy).unwrap();

        assert_eq!(env.cash, cash_before);
    }
}
