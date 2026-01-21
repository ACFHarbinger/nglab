/*!
 * Integration module for external accounts.
 *
 * Manages connections and credentials for external services like Polymarket.
 * Credentials are stored as JSON blobs in the secure Vault.
 */

use crate::secret::vault::VaultManager;
use serde::{Deserialize, Serialize};

/// Configuration for Polymarket integration.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PolymarketConfig {
    /// API key for authentication.
    pub api_key: String,
    /// API secret for signing requests.
    pub secret: String,
    /// Passphrase for API access.
    pub passphrase: String,
    /// Optional proxy address for connection.
    pub proxy_address: Option<String>,
}

/// Enum holding configuration for different integration services.
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "service", content = "config")]
pub enum IntegrationConfig {
    /// Polymarket service configuration.
    #[serde(rename = "polymarket")]
    Polymarket(PolymarketConfig),
}

/// Represents a stored external integration entry.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ExternalIntegration {
    /// Unique identifier from the vault.
    pub id: i64,
    /// Name of the service (e.g., "polymarket").
    pub service_name: String,
    /// The specific configuration for the service.
    pub config: IntegrationConfig,
    /// Creation timestamp.
    pub created_at: String,
}

/// Manages external service integrations backed by the vault.
pub struct IntegrationManager {
    vault: VaultManager,
}

impl IntegrationManager {
    /// Creates a new IntegrationManager using the provided VaultManager.
    pub fn new(vault: VaultManager) -> Self {
        Self { vault }
    }

    /// Save an integration configuration to the vault.
    pub fn save_integration(
        &self,
        key_str: &str,
        service: &str,
        config: IntegrationConfig,
    ) -> Result<i64, String> {
        let config_json = serde_json::to_string(&config).map_err(|e| e.to_string())?;
        let label = format!("integration:{}", service);

        // Use VaultManager to store the JSON blob
        self.vault
            .add_secret(key_str, &label, &config_json)
            .map_err(|e| e.to_string())
    }

    /// List all active integrations found in the vault.
    pub fn list_integrations(&self, key_str: &str) -> Result<Vec<ExternalIntegration>, String> {
        let secrets = self
            .vault
            .list_secrets(key_str)
            .map_err(|e| e.to_string())?;
        let mut integrations = Vec::new();

        for secret_summary in secrets {
            if secret_summary.label.starts_with("integration:") {
                let full_secret = self
                    .vault
                    .get_secret(key_str, secret_summary.id)
                    .map_err(|e| e.to_string())?
                    .ok_or_else(|| "Integration secret not found".to_string())?;

                match serde_json::from_str::<IntegrationConfig>(&full_secret.value) {
                    Ok(config) => {
                        integrations.push(ExternalIntegration {
                            id: full_secret.id,
                            service_name: secret_summary.label.replace("integration:", ""),
                            config,
                            created_at: full_secret.created_at,
                        });
                    }
                    Err(e) => {
                        eprintln!(
                            "⚠️ Failed to parse integration config for {}: {}",
                            secret_summary.label, e
                        );
                    }
                }
            }
        }

        Ok(integrations)
    }

    /// Delete an integration from the vault.
    pub fn delete_integration(&self, key_str: &str, id: i64) -> Result<(), String> {
        self.vault
            .delete_secret(key_str, id)
            .map_err(|e| e.to_string())
    }
}
