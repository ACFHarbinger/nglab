/*!
 * Tauri commands for the SQLCipher encrypted vault.
 */

use nglab::secret::vault::{VaultEntry, VaultManager, VaultSummary};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
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

#[derive(Debug, Serialize, Deserialize)]
pub struct VaultResponse<T> {
    pub success: bool,
    pub message: String,
    pub data: Option<T>,
}

fn get_vault_path(_app: &AppHandle) -> Result<PathBuf, String> {
    // Relocate to assets/secrets in the project root
    let mut path = PathBuf::from("/home/pkhunter/Repositories/nglab");
    path.push("assets");
    path.push("secrets");

    // Ensure the directory exists
    if !path.exists() {
        fs::create_dir_all(&path).map_err(|e| e.to_string())?;
    }

    path.push("vault.db");
    Ok(path)
}

/// Unlock the vault by deriving the master key
#[tauri::command]
pub async fn unlock_vault(
    app: AppHandle,
    password: String,
    state: State<'_, VaultState>,
) -> Result<VaultResponse<()>, String> {
    let salt = b"nglab-vault-salt-2026";
    let key_str = VaultManager::derive_key_string(&password, salt);

    let path = get_vault_path(&app)?;
    let manager = VaultManager::new(path);

    // Test open and initialize
    if let Err(e) = manager.init_db(&key_str) {
        return Ok(VaultResponse {
            success: false,
            message: format!("Failed to unlock or initialize vault: {}", e),
            data: None,
        });
    }

    if let Ok(mut master_key) = state.master_key.lock() {
        *master_key = Some(key_str);
    }

    Ok(VaultResponse {
        success: true,
        message: "Vault unlocked".to_string(),
        data: None,
    })
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
    app: AppHandle,
    label: String,
    value: String,
    state: State<'_, VaultState>,
) -> Result<VaultResponse<i64>, String> {
    let master_key = if let Ok(key_guard) = state.master_key.lock() {
        match key_guard.as_ref() {
            Some(key) => key.clone(),
            None => {
                return Ok(VaultResponse {
                    success: false,
                    message: "Vault is locked".to_string(),
                    data: None,
                })
            }
        }
    } else {
        return Err("Failed to lock state".to_string());
    };

    let path = get_vault_path(&app)?;
    let manager = VaultManager::new(path);
    match manager.add_secret(&master_key, &label, &value) {
        Ok(id) => Ok(VaultResponse {
            success: true,
            message: "Secret added".to_string(),
            data: Some(id),
        }),
        Err(e) => Ok(VaultResponse {
            success: false,
            message: format!("Failed to add secret: {}", e),
            data: None,
        }),
    }
}

/// List all secrets in the vault
#[tauri::command]
pub async fn list_vault_secrets(
    app: AppHandle,
    state: State<'_, VaultState>,
) -> Result<VaultResponse<Vec<VaultSummary>>, String> {
    let master_key = if let Ok(key_guard) = state.master_key.lock() {
        match key_guard.as_ref() {
            Some(key) => key.clone(),
            None => {
                return Ok(VaultResponse {
                    success: false,
                    message: "Vault is locked".to_string(),
                    data: None,
                })
            }
        }
    } else {
        return Err("Failed to lock state".to_string());
    };

    let path = get_vault_path(&app)?;
    let manager = VaultManager::new(path);
    match manager.list_secrets(&master_key) {
        Ok(entries) => Ok(VaultResponse {
            success: true,
            message: "Secrets listed".to_string(),
            data: Some(entries),
        }),
        Err(e) => Ok(VaultResponse {
            success: false,
            message: format!("Failed to list secrets: {}", e),
            data: None,
        }),
    }
}

/// Get a secret (decrypted automatically by SQLCipher)
#[tauri::command]
pub async fn get_vault_secret(
    app: AppHandle,
    id: i64,
    state: State<'_, VaultState>,
) -> Result<VaultResponse<VaultEntry>, String> {
    let master_key = if let Ok(key_guard) = state.master_key.lock() {
        match key_guard.as_ref() {
            Some(key) => key.clone(),
            None => {
                return Ok(VaultResponse {
                    success: false,
                    message: "Vault is locked".to_string(),
                    data: None,
                })
            }
        }
    } else {
        return Err("Failed to lock state".to_string());
    };

    let path = get_vault_path(&app)?;
    let manager = VaultManager::new(path);
    match manager.get_secret(&master_key, id) {
        Ok(Some(entry)) => Ok(VaultResponse {
            success: true,
            message: "Secret retrieved".to_string(),
            data: Some(entry),
        }),
        Ok(None) => Ok(VaultResponse {
            success: false,
            message: "Secret not found".to_string(),
            data: None,
        }),
        Err(e) => Ok(VaultResponse {
            success: false,
            message: format!("Failed to retrieve secret: {}", e),
            data: None,
        }),
    }
}

/// Delete a secret
#[tauri::command]
pub async fn delete_vault_secret(
    app: AppHandle,
    id: i64,
    state: State<'_, VaultState>,
) -> Result<VaultResponse<()>, String> {
    let master_key = if let Ok(key_guard) = state.master_key.lock() {
        match key_guard.as_ref() {
            Some(key) => key.clone(),
            None => {
                return Ok(VaultResponse {
                    success: false,
                    message: "Vault is locked".to_string(),
                    data: None,
                })
            }
        }
    } else {
        return Err("Failed to lock state".to_string());
    };

    let path = get_vault_path(&app)?;
    let manager = VaultManager::new(path);
    match manager.delete_secret(&master_key, id) {
        Ok(()) => Ok(VaultResponse {
            success: true,
            message: "Secret deleted".to_string(),
            data: None,
        }),
        Err(e) => Ok(VaultResponse {
            success: false,
            message: format!("Failed to delete secret: {}", e),
            data: None,
        }),
    }
}
