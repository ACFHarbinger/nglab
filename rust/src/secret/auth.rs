/*!
 * Authentication module for NGLab.
 *
 * Provides secure password hashing with Argon2id and credential storage
 * using the OS-native keyring (Keychain on macOS, Credential Manager on Windows,
 * Secret Service on Linux).
 */

use argon2::{
    password_hash::{PasswordHash, PasswordHasher, PasswordVerifier, SaltString},
    Argon2,
};
use keyring::Entry;
use password_hash::rand_core::OsRng;
use serde::{Deserialize, Serialize};

/// Service name for keyring entries
const SERVICE_NAME: &str = "nglab";

/// Result type for authentication operations
pub type AuthResult<T> = Result<T, AuthError>;

/// Authentication errors
#[derive(Debug, thiserror::Error)]
pub enum AuthError {
    #[error("User not found: {0}")]
    UserNotFound(String),

    #[error("Invalid password")]
    InvalidPassword,

    #[error("User already exists: {0}")]
    UserAlreadyExists(String),

    #[error("Password hashing failed: {0}")]
    HashingError(String),

    #[error("Keyring error: {0}")]
    KeyringError(String),

    #[error("Serialization error: {0}")]
    SerializationError(String),
}

/// Stored user credentials
#[derive(Debug, Serialize, Deserialize)]
pub struct StoredCredential {
    /// Argon2id password hash (includes salt)
    pub password_hash: String,
    /// Account creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
}

/// Authentication manager
pub struct AuthManager;

impl AuthManager {
    /// Hash a password using Argon2id
    fn hash_password(password: &str) -> AuthResult<String> {
        let salt = SaltString::generate(&mut OsRng);
        let argon2 = Argon2::default();

        argon2
            .hash_password(password.as_bytes(), &salt)
            .map(|hash| hash.to_string())
            .map_err(|e| AuthError::HashingError(e.to_string()))
    }

    /// Verify a password against a stored hash
    fn verify_password(password: &str, password_hash: &str) -> AuthResult<bool> {
        let parsed_hash =
            PasswordHash::new(password_hash).map_err(|e| AuthError::HashingError(e.to_string()))?;

        Ok(Argon2::default()
            .verify_password(password.as_bytes(), &parsed_hash)
            .is_ok())
    }

    /// Get keyring entry for a username
    fn get_entry(username: &str) -> AuthResult<Entry> {
        Entry::new(SERVICE_NAME, username).map_err(|e| AuthError::KeyringError(e.to_string()))
    }

    /// Create a new user account
    ///
    /// Hashes the password with Argon2id and stores it in the OS keyring.
    pub fn create_account(username: &str, password: &str) -> AuthResult<()> {
        let entry = Self::get_entry(username)?;

        // Check if user already exists
        match entry.get_password() {
            Ok(_) => return Err(AuthError::UserAlreadyExists(username.to_string())),
            Err(keyring::Error::NoEntry) => { /* User does not exist, proceed */ }
            Err(e) => {
                return Err(AuthError::KeyringError(format!(
                    "Error checking existing user: {}",
                    e
                )))
            }
        }

        // Hash password
        let password_hash = Self::hash_password(password)?;

        // Create credential
        let credential = StoredCredential {
            password_hash,
            created_at: chrono::Utc::now(),
        };

        // Serialize and store
        let credential_json = serde_json::to_string(&credential)
            .map_err(|e| AuthError::SerializationError(e.to_string()))?;

        entry
            .set_password(&credential_json)
            .map_err(|e| AuthError::KeyringError(format!("Failed to save credential: {}", e)))?;

        Ok(())
    }

    /// Verify login credentials
    ///
    /// Retrieves the stored hash from keyring and verifies the password.
    pub fn login(username: &str, password: &str) -> AuthResult<bool> {
        let entry = Self::get_entry(username)?;

        // Get stored credential
        let credential_json = match entry.get_password() {
            Ok(json) => json,
            Err(keyring::Error::NoEntry) => {
                return Err(AuthError::UserNotFound(username.to_string()))
            }
            Err(e) => {
                return Err(AuthError::KeyringError(format!(
                    "Failed to retrieve credential: {}",
                    e
                )))
            }
        };

        let credential: StoredCredential = serde_json::from_str(&credential_json)
            .map_err(|e| AuthError::SerializationError(e.to_string()))?;

        // Verify password
        Self::verify_password(password, &credential.password_hash)
    }

    /// Delete a user account
    pub fn delete_account(username: &str) -> AuthResult<()> {
        let entry = Self::get_entry(username)?;

        entry
            .delete_credential()
            .map_err(|e| AuthError::KeyringError(e.to_string()))?;

        Ok(())
    }

    /// Check if a user exists
    pub fn user_exists(username: &str) -> AuthResult<bool> {
        let entry = Self::get_entry(username)?;
        Ok(entry.get_password().is_ok())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use password_hash::rand_core::RngCore;

    #[test]
    fn test_password_hashing() {
        let password = "super_secret_123";
        let hash = AuthManager::hash_password(password).unwrap();

        // Hash should be a valid Argon2 hash string
        assert!(hash.starts_with("$argon2"));

        // Verify should succeed
        assert!(AuthManager::verify_password(password, &hash).unwrap());

        // Wrong password should fail
        assert!(!AuthManager::verify_password("wrong_password", &hash).unwrap());
    }

    #[test]
    fn test_full_auth_flow() {
        let username = format!("test_user_{}", OsRng.next_u64());
        let password = "test_password_123";

        // Create account
        AuthManager::create_account(&username, &password).expect("Failed to create account");

        // Login should succeed
        assert!(AuthManager::login(&username, &password).expect("Failed to login"));

        // Login with wrong password should return false (invalid password)
        assert!(
            !AuthManager::login(&username, "wrong_pass").expect("Failed to check wrong password")
        );

        // Clean up
        AuthManager::delete_account(&username).expect("Failed to delete account");
    }
}
