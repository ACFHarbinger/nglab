/*!
 * Tauri commands for external service integrations.
 */

use crate::commands::vault::VaultState;
use nglab::security::integrations::{
    ExternalIntegration, IntegrationConfig, IntegrationManager, PolymarketConfig,
};
use nglab::security::vault::{VaultManager, VaultResponse};
use nglab::web::polymarket::{EventResponse, MarketSearchResult, PolymarketScraper};
use tauri::State;

/// Save a Polymarket integration configuration.
#[tauri::command]
pub async fn save_polymarket_integration(
    api_key: String,
    secret: String,
    passphrase: String,
    proxy_address: Option<String>,
    state: State<'_, VaultState>,
) -> Result<VaultResponse<i64>, String> {
    let master_key = if let Ok(key_guard) = state.master_key.lock() {
        match key_guard.as_ref() {
            Some(key) => key.clone(),
            None => return Ok(VaultResponse::error("Vault is locked")),
        }
    } else {
        return Err("Failed to lock vault state".to_string());
    };

    let vault_manager = VaultManager::with_default_path().map_err(|e| e.to_string())?;
    let integration_manager = IntegrationManager::new(vault_manager);

    let config = IntegrationConfig::Polymarket(PolymarketConfig {
        api_key,
        secret,
        passphrase,
        proxy_address,
    });

    match integration_manager.save_integration(&master_key, "polymarket", config) {
        Ok(id) => Ok(VaultResponse::success(
            "Polymarket integration saved",
            Some(id),
        )),
        Err(e) => Ok(VaultResponse::error(&format!(
            "Failed to save integration: {}",
            e
        ))),
    }
}

/// List all active integrations.
#[tauri::command]
pub async fn list_integrations(
    state: State<'_, VaultState>,
) -> Result<VaultResponse<Vec<ExternalIntegration>>, String> {
    let master_key = if let Ok(key_guard) = state.master_key.lock() {
        match key_guard.as_ref() {
            Some(key) => key.clone(),
            None => return Ok(VaultResponse::error("Vault is locked")),
        }
    } else {
        return Err("Failed to lock vault state".to_string());
    };

    let vault_manager = VaultManager::with_default_path().map_err(|e| e.to_string())?;
    let integration_manager = IntegrationManager::new(vault_manager);

    match integration_manager.list_integrations(&master_key) {
        Ok(integrations) => Ok(VaultResponse::success(
            "Integrations listed",
            Some(integrations),
        )),
        Err(e) => Ok(VaultResponse::error(&format!(
            "Failed to list integrations: {}",
            e
        ))),
    }
}

/// Delete an integration by ID.
#[tauri::command]
pub async fn delete_integration(
    id: i64,
    state: State<'_, VaultState>,
) -> Result<VaultResponse<()>, String> {
    let master_key = if let Ok(key_guard) = state.master_key.lock() {
        match key_guard.as_ref() {
            Some(key) => key.clone(),
            None => return Ok(VaultResponse::error("Vault is locked")),
        }
    } else {
        return Err("Failed to lock vault state".to_string());
    };

    let vault_manager = VaultManager::with_default_path().map_err(|e| e.to_string())?;
    let integration_manager = IntegrationManager::new(vault_manager);

    match integration_manager.delete_integration(&master_key, id) {
        Ok(_) => Ok(VaultResponse::success("Integration deleted", None)),
        Err(e) => Ok(VaultResponse::error(&format!(
            "Failed to delete integration: {}",
            e
        ))),
    }
}

/// Get trending markets from Polymarket (requires integration).
#[tauri::command]
pub async fn get_trending_markets(
    limit: usize,
    state: State<'_, VaultState>,
) -> Result<VaultResponse<Vec<EventResponse>>, String> {
    let master_key = if let Ok(key_guard) = state.master_key.lock() {
        match key_guard.as_ref() {
            Some(key) => key.clone(),
            None => return Ok(VaultResponse::error("Vault is locked")),
        }
    } else {
        return Err("Failed to lock vault state".to_string());
    };

    let vault_manager = VaultManager::with_default_path().map_err(|e| e.to_string())?;
    let integration_manager = IntegrationManager::new(vault_manager);

    // Verify integration exists
    let integrations = integration_manager
        .list_integrations(&master_key)
        .map_err(|e| e.to_string())?;

    let has_polymarket = integrations.iter().any(|i| i.service_name == "polymarket");

    if !has_polymarket {
        return Ok(VaultResponse::error("Polymarket account not connected"));
    }

    let scraper = PolymarketScraper::new();
    match scraper.get_trending_events(limit).await {
        Ok(events) => Ok(VaultResponse::success(
            "Trending markets fetched",
            Some(events),
        )),
        Err(e) => Ok(VaultResponse::error(&format!(
            "Failed to fetch trending markets: {}",
            e
        ))),
    }
}

/// Search for open Polymarket markets by query string.
#[tauri::command]
pub async fn search_polymarket_markets(
    query: String,
    limit: usize,
    state: State<'_, VaultState>,
) -> Result<VaultResponse<Vec<MarketSearchResult>>, String> {
    let master_key = if let Ok(key_guard) = state.master_key.lock() {
        match key_guard.as_ref() {
            Some(key) => key.clone(),
            None => return Ok(VaultResponse::error("Vault is locked")),
        }
    } else {
        return Err("Failed to lock vault state".to_string());
    };

    let vault_manager = VaultManager::with_default_path().map_err(|e| e.to_string())?;
    let integration_manager = IntegrationManager::new(vault_manager);

    // Verify integration exists
    let integrations = integration_manager
        .list_integrations(&master_key)
        .map_err(|e| e.to_string())?;

    let has_polymarket = integrations.iter().any(|i| i.service_name == "polymarket");

    if !has_polymarket {
        return Ok(VaultResponse::error("Polymarket account not connected"));
    }

    let scraper = PolymarketScraper::new();
    match scraper.search_markets(&query, limit).await {
        Ok(results) => Ok(VaultResponse::success(
            &format!("Found {} markets", results.len()),
            Some(results),
        )),
        Err(e) => Ok(VaultResponse::error(&format!(
            "Failed to search markets: {}",
            e
        ))),
    }
}

/// Get open/active Polymarket markets for browsing (requires integration).
#[tauri::command]
pub async fn get_open_polymarket_markets(
    limit: usize,
    state: State<'_, VaultState>,
) -> Result<VaultResponse<Vec<MarketSearchResult>>, String> {
    let master_key = if let Ok(key_guard) = state.master_key.lock() {
        match key_guard.as_ref() {
            Some(key) => key.clone(),
            None => return Ok(VaultResponse::error("Vault is locked")),
        }
    } else {
        return Err("Failed to lock vault state".to_string());
    };

    let vault_manager = VaultManager::with_default_path().map_err(|e| e.to_string())?;
    let integration_manager = IntegrationManager::new(vault_manager);

    // Verify integration exists
    let integrations = integration_manager
        .list_integrations(&master_key)
        .map_err(|e| e.to_string())?;

    let has_polymarket = integrations.iter().any(|i| i.service_name == "polymarket");

    if !has_polymarket {
        return Ok(VaultResponse::error("Polymarket account not connected"));
    }

    let scraper = PolymarketScraper::new();
    match scraper.get_open_markets(limit).await {
        Ok(results) => Ok(VaultResponse::success(
            &format!("Found {} open markets", results.len()),
            Some(results),
        )),
        Err(e) => Ok(VaultResponse::error(&format!(
            "Failed to fetch open markets: {}",
            e
        ))),
    }
}

/// Get public Polymarket markets without requiring authentication.
/// This uses the public gamma API which doesn't require credentials.
#[tauri::command]
pub async fn get_public_polymarket_markets(
    limit: usize,
) -> Result<VaultResponse<Vec<MarketSearchResult>>, String> {
    let scraper = PolymarketScraper::new();
    match scraper.get_open_markets(limit).await {
        Ok(results) => Ok(VaultResponse::success(
            &format!("Found {} markets", results.len()),
            Some(results),
        )),
        Err(e) => Ok(VaultResponse::error(&format!(
            "Failed to fetch markets: {}",
            e
        ))),
    }
}

/// Search public Polymarket markets without requiring authentication.
#[tauri::command]
pub async fn search_public_polymarket_markets(
    query: String,
    limit: usize,
) -> Result<VaultResponse<Vec<MarketSearchResult>>, String> {
    let scraper = PolymarketScraper::new();
    match scraper.search_markets(&query, limit).await {
        Ok(results) => Ok(VaultResponse::success(
            &format!("Found {} markets", results.len()),
            Some(results),
        )),
        Err(e) => Ok(VaultResponse::error(&format!(
            "Failed to search markets: {}",
            e
        ))),
    }
}
