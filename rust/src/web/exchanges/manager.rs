use super::binance::Binance;
use super::deribit::Deribit;
use super::kraken::Kraken;
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

        // Register Kraken
        exchanges.insert("Kraken".to_string(), Arc::new(Mutex::new(Kraken::new())));

        // Register Deribit
        exchanges.insert("Deribit".to_string(), Arc::new(Mutex::new(Deribit::new())));

        Self {
            exchanges,
            active_exchange_name: "Polymarket".to_string(), // Default
        }
    }

    pub fn list_exchanges(&self) -> Vec<String> {
        let mut keys: Vec<String> = self.exchanges.keys().cloned().collect();
        keys.sort();
        keys
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

    pub fn get_all_exchanges(&self) -> HashMap<String, Arc<Mutex<dyn Exchange>>> {
        self.exchanges.clone()
    }

    /// Connect all exchanges that have stored credentials in the vault.
    pub async fn connect_all_from_vault(&self, master_key: &str) -> Result<(), String> {
        use crate::security::integrations::{IntegrationConfig, IntegrationManager};
        use crate::security::vault::VaultManager;

        let vault_manager = VaultManager::with_default_path().map_err(|e| e.to_string())?;
        let integration_manager = IntegrationManager::new(vault_manager);

        let integrations = integration_manager.list_integrations(master_key)?;

        for integration in integrations {
            let exchange_name = match &integration.config {
                IntegrationConfig::Binance(_) => "Binance",
                IntegrationConfig::Kraken(_) => "Kraken",
                IntegrationConfig::Deribit(_) => "Deribit",
                IntegrationConfig::Polymarket(_) => "Polymarket",
            };

            if let Some(exchange_mutex) = self.exchanges.get(exchange_name) {
                let mut exchange = exchange_mutex.lock().await;

                let (key, secret) = match &integration.config {
                    IntegrationConfig::Polymarket(c) => {
                        (Some(c.api_key.clone()), Some(c.secret.clone()))
                    }
                    IntegrationConfig::Binance(c) => {
                        (Some(c.api_key.clone()), Some(c.secret.clone()))
                    }
                    IntegrationConfig::Kraken(c) => {
                        (Some(c.api_key.clone()), Some(c.secret.clone()))
                    }
                    IntegrationConfig::Deribit(c) => {
                        (Some(c.api_key.clone()), Some(c.secret.clone()))
                    }
                };

                if let Err(e) = exchange.connect(key, secret).await {
                    tracing::error!("Failed to connect exchange {}: {}", exchange_name, e);
                } else {
                    tracing::info!(
                        "Successfully connected {} using vaulted credentials",
                        exchange_name
                    );
                }
            }
        }

        Ok(())
    }
}

impl Default for ExchangeManager {
    fn default() -> Self {
        Self::new()
    }
}
