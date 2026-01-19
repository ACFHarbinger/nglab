use crate::errors::ArenaError;
use crate::simulation::gym::ActionType;
use crate::simulation::orderbook::{OrderBook, Side};
#[cfg(feature = "python")]
use numpy::{PyArray2, ToPyArray};
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyDict;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::collections::HashMap;

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

    // Step function for Rust usage (if needed)
    // ...
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

        // Generate observation
        let obs_data = self.generate_observation_data();
        let total_features = self.assets.len() * self.features_per_asset;

        let obs_array = ndarray::Array2::from_shape_vec((self.lookback, total_features), obs_data)
            .map_err(|e| ArenaError::DataLoading(format!("Invalid shape: {}", e)))?;

        let info = PyDict::new(py);
        Ok((obs_array.to_pyarray(py), info.into()))
    }

    // Actions: List of integers (one per asset)
    pub fn step<'py>(
        &mut self,
        py: Python<'py>,
        actions: Vec<i32>,
    ) -> PyResult<(Bound<'py, PyArray2<f64>>, f64, bool, bool, Py<PyAny>)> {
        let prev_val = self.portfolio_value();

        // Execute actions per asset
        for (i, &action_idx) in actions.iter().enumerate() {
            if i >= self.assets.len() {
                break;
            }
            let action = ActionType::from(action_idx);

            self.execute_asset_action(i, action);
        }

        self.current_step += 1;
        self.total_steps += 1;

        let new_val = self.portfolio_value();
        let returns = if prev_val > 0.0 {
            (new_val - prev_val) / prev_val
        } else {
            0.0
        };
        let reward = returns * 100.0; // Simple reward

        let terminated = new_val <= 0.0; // Bankruptcy
                                         // Truncated if max steps reached
                                         // We assume equal length price series for simplified truncated check
        let truncated = self.total_steps as usize >= self.max_steps;

        let obs_data = self.generate_observation_data();
        let total_features = self.assets.len() * self.features_per_asset;

        let obs_array = ndarray::Array2::from_shape_vec((self.lookback, total_features), obs_data)
            .map_err(|e| ArenaError::DataLoading(format!("Invalid shape: {}", e)))?;

        let info = PyDict::new(py);
        info.set_item("portfolio_value", new_val)?;
        info.set_item("cash", self.cash)?;

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
    fn execute_asset_action(&mut self, asset: &str, action: ActionType) {
        // Simplified execution: 10% of current CASH per trade? Or 10% of Capital?
        // Let's us fixed share of initial capital for simplicity per asset trade
        let trade_size = self.initial_capital * 0.05;

        let price = match self
            .prices
            .get(asset)
            .and_then(|v| v.get(self.current_step))
        {
            Some(&p) => p,
            None => return,
        };

        match action {
            ActionType::Hold => {}
            ActionType::Buy => {
                let cost_with_fee = trade_size * (1.0 + self.transaction_cost);
                if self.cash >= cost_with_fee {
                    let shares = trade_size / price;
                    self.cash -= cost_with_fee;
                    *self.positions.get_mut(asset).unwrap() += shares;
                    // Update orderbook (assumed)
                    if let Some(ob) = self.orderbooks.get_mut(asset) {
                        ob.submit_market_order(shares, Side::Bid);
                    }
                }
            }
            ActionType::Sell => {
                let pos = *self.positions.get(asset).unwrap();
                if pos > 0.0 {
                    let sell_shares = (trade_size / price).min(pos);
                    let proceeds = sell_shares * price;
                    let fee = proceeds * self.transaction_cost;
                    self.cash += proceeds - fee;
                    *self.positions.get_mut(asset).unwrap() -= sell_shares;
                    if let Some(ob) = self.orderbooks.get_mut(asset) {
                        ob.submit_market_order(sell_shares, Side::Ask);
                    }
                }
            }
        }
    }

    fn generate_observation_data(&self) -> Vec<f64> {
        // Simple concatenation of features for all assets
        // (lookback, assets * features)
        let total_features = self.assets.len() * self.features_per_asset;
        let mut data = vec![0.0; self.lookback * total_features];

        // This is inefficient (row-major filling logic needed).
        // Gym usually expects (Time, Features).
        // Features = [Asset1_Price, Asset1_Ret... Asset2_Price...]

        for t in 0..self.lookback {
            let step_idx = self
                .current_step
                .saturating_sub(self.lookback)
                .saturating_add(t);
            let row_start = t * total_features;

            for (i, asset) in self.assets.iter().enumerate() {
                let asset_offset = i * self.features_per_asset;
                let idx = row_start + asset_offset;

                let price = self
                    .prices
                    .get(asset)
                    .and_then(|v| v.get(step_idx))
                    .unwrap_or(&0.0);
                // ... calc returns ...

                data[idx] = *price; // Placeholder logic
                                    // ... fill other features
            }
        }
        data
    }
}
