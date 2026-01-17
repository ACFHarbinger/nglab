/*!
 * Tauri commands for user authentication.
 */

use nglab::secret::auth::{AuthError, AuthManager, AuthResponse};
use std::sync::Mutex;
use tauri::State;

/// Session state for tracking logged-in user
pub struct AuthState {
    pub current_user: Mutex<Option<String>>,
}

impl Default for AuthState {
    fn default() -> Self {
        Self {
            current_user: Mutex::new(None),
        }
    }
}

/// Create a new user account
#[tauri::command]
pub async fn create_account(username: String, password: String) -> Result<AuthResponse, String> {
    // Validate inputs
    if username.trim().is_empty() {
        return Ok(AuthResponse::error("Username cannot be empty"));
    }
    if password.len() < 8 {
        return Ok(AuthResponse::error(
            "Password must be at least 8 characters",
        ));
    }

    match AuthManager::create_account(&username, &password) {
        Ok(()) => Ok(AuthResponse::success(
            "Account created successfully",
            Some(username),
        )),
        Err(AuthError::UserAlreadyExists(_)) => Ok(AuthResponse::error("Username already exists")),
        Err(e) => Ok(AuthResponse::error(&format!(
            "Failed to create account: {}",
            e
        ))),
    }
}

/// Login with username and password
#[tauri::command]
pub async fn login(
    username: String,
    password: String,
    state: State<'_, AuthState>,
) -> Result<AuthResponse, String> {
    match AuthManager::login(&username, &password) {
        Ok(true) => {
            // Set current user in session state
            if let Ok(mut current_user) = state.current_user.lock() {
                *current_user = Some(username.clone());
            }
            Ok(AuthResponse::success("Login successful", Some(username)))
        }
        Ok(false) => Ok(AuthResponse::error("Invalid password")),
        Err(AuthError::UserNotFound(_)) => Ok(AuthResponse::error("User not found")),
        Err(e) => Ok(AuthResponse::error(&format!("Login failed: {}", e))),
    }
}

/// Logout current user
#[tauri::command]
pub async fn logout(state: State<'_, AuthState>) -> Result<AuthResponse, String> {
    if let Ok(mut current_user) = state.current_user.lock() {
        *current_user = None;
    }
    Ok(AuthResponse::success("Logged out successfully", None))
}

/// Get current logged-in user
#[tauri::command]
pub async fn get_current_user(state: State<'_, AuthState>) -> Result<AuthResponse, String> {
    if let Ok(current_user) = state.current_user.lock() {
        if let Some(username) = current_user.clone() {
            return Ok(AuthResponse::success("User is logged in", Some(username)));
        }
    }
    Ok(AuthResponse::error("No user logged in"))
}
