/*!
 * Tauri commands for the SQLCipher encrypted vault.
 */

use nglab::secret::vault::{VaultEntry, VaultFavorite, VaultManager, VaultResponse, VaultSummary};
use std::sync::Mutex;
use tauri::{AppHandle, State};

/// State for the vault, holding the derived master key for the session
pub struct VaultState {
    pub master_key: Mutex<Option<String>>,
}

impl Default for VaultState {
    fn default() -> Self {
        Self {
            master_key: Mutex::new(None),
        }
    }
}

/// Unlock the vault by deriving the master key
#[tauri::command]
pub async fn unlock_vault(
    _app: AppHandle,
    password: String,
    state: State<'_, VaultState>,
) -> Result<VaultResponse<()>, String> {
    let salt = b"nglab-vault-salt-2026";
    let key_str = VaultManager::derive_key_string(&password, salt);

    let manager = VaultManager::with_default_path()?;

    // Test open and initialize
    if let Err(e) = manager.init_db(&key_str) {
        return Ok(VaultResponse::error(&format!(
            "Failed to unlock or initialize vault: {}",
            e
        )));
    }

    if let Ok(mut master_key) = state.master_key.lock() {
        *master_key = Some(key_str);
    }

    Ok(VaultResponse::success("Vault unlocked", None))
}

/// Check if vault is unlocked
#[tauri::command]
pub async fn is_vault_unlocked(state: State<'_, VaultState>) -> Result<bool, String> {
    if let Ok(master_key) = state.master_key.lock() {
        return Ok(master_key.is_some());
    }
    Ok(false)
}

/// Lock the vault (clear the key)
#[tauri::command]
pub async fn lock_vault(state: State<'_, VaultState>) -> Result<(), String> {
    if let Ok(mut master_key) = state.master_key.lock() {
        *master_key = None;
    }
    Ok(())
}

/// Add a secret to the vault
#[tauri::command]
pub async fn add_vault_secret(
    _app: AppHandle,
    label: String,
    value: String,
    state: State<'_, VaultState>,
) -> Result<VaultResponse<i64>, String> {
    let master_key = if let Ok(key_guard) = state.master_key.lock() {
        match key_guard.as_ref() {
            Some(key) => key.clone(),
            None => return Ok(VaultResponse::error("Vault is locked")),
        }
    } else {
        return Err("Failed to lock state".to_string());
    };

    let manager = VaultManager::with_default_path()?;
    match manager.add_secret(&master_key, &label, &value) {
        Ok(id) => Ok(VaultResponse::success("Secret added", Some(id))),
        Err(e) => Ok(VaultResponse::error(&format!(
            "Failed to add secret: {}",
            e
        ))),
    }
}

/// List all secrets in the vault
#[tauri::command]
pub async fn list_vault_secrets(
    _app: AppHandle,
    state: State<'_, VaultState>,
) -> Result<VaultResponse<Vec<VaultSummary>>, String> {
    let master_key = if let Ok(key_guard) = state.master_key.lock() {
        match key_guard.as_ref() {
            Some(key) => key.clone(),
            None => return Ok(VaultResponse::error("Vault is locked")),
        }
    } else {
        return Err("Failed to lock state".to_string());
    };

    let manager = VaultManager::with_default_path()?;
    match manager.list_secrets(&master_key) {
        Ok(entries) => Ok(VaultResponse::success("Secrets listed", Some(entries))),
        Err(e) => Ok(VaultResponse::error(&format!(
            "Failed to list secrets: {}",
            e
        ))),
    }
}

/// Get a secret (decrypted automatically by SQLCipher)
#[tauri::command]
pub async fn get_vault_secret(
    _app: AppHandle,
    id: i64,
    state: State<'_, VaultState>,
) -> Result<VaultResponse<VaultEntry>, String> {
    let master_key = if let Ok(key_guard) = state.master_key.lock() {
        match key_guard.as_ref() {
            Some(key) => key.clone(),
            None => return Ok(VaultResponse::error("Vault is locked")),
        }
    } else {
        return Err("Failed to lock state".to_string());
    };

    let manager = VaultManager::with_default_path()?;
    match manager.get_secret(&master_key, id) {
        Ok(Some(entry)) => Ok(VaultResponse::success("Secret retrieved", Some(entry))),
        Ok(None) => Ok(VaultResponse::error("Secret not found")),
        Err(e) => Ok(VaultResponse::error(&format!(
            "Failed to retrieve secret: {}",
            e
        ))),
    }
}

/// Delete a secret
#[tauri::command]
pub async fn delete_vault_secret(
    _app: AppHandle,
    id: i64,
    state: State<'_, VaultState>,
) -> Result<VaultResponse<()>, String> {
    let master_key = if let Ok(key_guard) = state.master_key.lock() {
        match key_guard.as_ref() {
            Some(key) => key.clone(),
            None => return Ok(VaultResponse::error("Vault is locked")),
        }
    } else {
        return Err("Failed to lock state".to_string());
    };

    let manager = VaultManager::with_default_path()?;
    match manager.delete_secret(&master_key, id) {
        Ok(()) => Ok(VaultResponse::success("Secret deleted", None)),
        Err(e) => Ok(VaultResponse::error(&format!(
            "Failed to delete secret: {}",
            e
        ))),
    }
}

/// Add a favorite market to the vault
#[tauri::command]
pub async fn add_favorite(
    _app: AppHandle,
    id: String,
    symbol: String,
    name: String,
    metadata_json: String,
    state: State<'_, VaultState>,
) -> Result<VaultResponse<()>, String> {
    let master_key = if let Ok(key_guard) = state.master_key.lock() {
        match key_guard.as_ref() {
            Some(key) => key.clone(),
            None => return Ok(VaultResponse::error("Vault is locked")),
        }
    } else {
        return Err("Failed to lock state".to_string());
    };

    let manager = VaultManager::with_default_path()?;
    match manager.add_favorite(&master_key, &id, &symbol, &name, &metadata_json) {
        Ok(()) => Ok(VaultResponse::success("Favorite added", None)),
        Err(e) => Ok(VaultResponse::error(&format!(
            "Failed to add favorite: {}",
            e
        ))),
    }
}

/// List all favorite markets
#[tauri::command]
pub async fn get_favorites(
    _app: AppHandle,
    state: State<'_, VaultState>,
) -> Result<VaultResponse<Vec<VaultFavorite>>, String> {
    let master_key = if let Ok(key_guard) = state.master_key.lock() {
        match key_guard.as_ref() {
            Some(key) => key.clone(),
            None => return Ok(VaultResponse::error("Vault is locked")),
        }
    } else {
        return Err("Failed to lock state".to_string());
    };

    let manager = VaultManager::with_default_path()?;
    match manager.list_favorites(&master_key) {
        Ok(favorites) => Ok(VaultResponse::success("Favorites listed", Some(favorites))),
        Err(e) => Ok(VaultResponse::error(&format!(
            "Failed to list favorites: {}",
            e
        ))),
    }
}

/// Remove a favorite market
#[tauri::command]
pub async fn remove_favorite(
    _app: AppHandle,
    id: String,
    state: State<'_, VaultState>,
) -> Result<VaultResponse<()>, String> {
    let master_key = if let Ok(key_guard) = state.master_key.lock() {
        match key_guard.as_ref() {
            Some(key) => key.clone(),
            None => return Ok(VaultResponse::error("Vault is locked")),
        }
    } else {
        return Err("Failed to lock state".to_string());
    };

    let manager = VaultManager::with_default_path()?;
    match manager.remove_favorite(&master_key, &id) {
        Ok(()) => Ok(VaultResponse::success("Favorite removed", None)),
        Err(e) => Ok(VaultResponse::error(&format!(
            "Failed to remove favorite: {}",
            e
        ))),
    }
}
