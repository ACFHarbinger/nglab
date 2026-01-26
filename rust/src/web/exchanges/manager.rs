use super::binance::Binance;
use super::polymarket_adapter::PolymarketAdapter;
use super::Exchange;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

pub struct ExchangeManager {
    exchanges: HashMap<String, Arc<Mutex<dyn Exchange>>>,
    active_exchange_name: String,
}

impl ExchangeManager {
    pub fn new() -> Self {
        let mut exchanges: HashMap<String, Arc<Mutex<dyn Exchange>>> = HashMap::new();

        // Register Binance
        exchanges.insert("Binance".to_string(), Arc::new(Mutex::new(Binance::new())));

        // Register Polymarket
        exchanges.insert(
            "Polymarket".to_string(),
            Arc::new(Mutex::new(PolymarketAdapter::new())),
        );

        Self {
            exchanges,
            active_exchange_name: "Polymarket".to_string(), // Default
        }
    }

    pub fn list_exchanges(&self) -> Vec<String> {
        self.exchanges.keys().cloned().collect()
    }

    pub fn set_active_exchange(&mut self, name: &str) -> Result<(), String> {
        if self.exchanges.contains_key(name) {
            self.active_exchange_name = name.to_string();
            Ok(())
        } else {
            Err(format!("Exchange {} not found", name))
        }
    }

    pub fn get_active_exchange(&self) -> Arc<Mutex<dyn Exchange>> {
        self.exchanges
            .get(&self.active_exchange_name)
            .unwrap()
            .clone()
    }

    pub fn get_active_name(&self) -> String {
        self.active_exchange_name.clone()
    }
}

impl Default for ExchangeManager {
    fn default() -> Self {
        Self::new()
    }
}
