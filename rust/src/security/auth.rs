use argon2::{
    password_hash::{PasswordHash, PasswordHasher, PasswordVerifier, SaltString},
    Argon2,
};
use password_hash::rand_core::OsRng;
use rusqlite::{params, Connection, Result as SqlResult};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

/// Standard response wrapper for authentication operations
#[derive(Debug, Serialize, Deserialize)]
pub struct AuthResponse {
    /// Whether the operation was successful.
    pub success: bool,
    /// Detailed message about the operation result.
    pub message: String,
    /// The authenticated username, if applicable.
    pub username: Option<String>,
}

impl AuthResponse {
    /// Creates a successful authentication response.
    pub fn success(message: &str, username: Option<String>) -> Self {
        Self {
            success: true,
            message: message.to_string(),
            username,
        }
    }

    /// Creates an error authentication response.
    pub fn error(message: &str) -> Self {
        Self {
            success: false,
            message: message.to_string(),
            username: None,
        }
    }
}

/// Result type for authentication operations
pub type AuthResult<T> = Result<T, AuthError>;

/// Authentication errors
#[derive(Debug, thiserror::Error)]
pub enum AuthError {
    /// User was not found in the storage.
    #[error("User not found: {0}")]
    UserNotFound(String),

    /// Password provided does not match the stored hash.
    #[error("Invalid password")]
    InvalidPassword,

    /// Attempted to create a user that already exists.
    #[error("User already exists: {0}")]
    UserAlreadyExists(String),

    /// Errors occurring during Argon2 hashing.
    #[error("Password hashing failed: {0}")]
    HashingError(String),

    /// Database errors.
    #[error("Database error: {0}")]
    DatabaseError(String),

    /// Internal data serialization failures.
    #[error("Serialization error: {0}")]
    SerializationError(String),
}

/// Stored user credentials
#[derive(Debug, Serialize, Deserialize)]
pub struct StoredCredential {
    /// Argon2id password hash (includes the salt).
    pub password_hash: String,
    /// The time when the credentials were created.
    pub created_at: chrono::DateTime<chrono::Utc>,
}

/// Manages credential storage using SQLite/SQLCipher.
pub struct CredentialManager {
    db_path: PathBuf,
}

impl CredentialManager {
    /// Default encryption key for credentials database (hashes are already secure)
    const DEFAULT_KEY: &'static str = "nglab_credentials_secure_v1";

    /// Creates a new CredentialManager with the given database path.
    pub fn new(db_path: PathBuf) -> Self {
        Self { db_path }
    }

    /// Gets the default path for the credentials database in `assets/secrets/credentials.db`.
    pub fn get_default_path() -> AuthResult<PathBuf> {
        let mut path = PathBuf::from("/home/pkhunter/Repositories/nglab");
        path.push("assets");
        path.push("secrets");

        if !path.exists() {
            fs::create_dir_all(&path).map_err(|e| AuthError::DatabaseError(e.to_string()))?;
        }

        path.push("credentials.db");
        Ok(path)
    }

    /// Creates a new CredentialManager using the default database path.
    pub fn with_default_path() -> AuthResult<Self> {
        Ok(Self::new(Self::get_default_path()?))
    }

    /// Opens an encrypted connection to the database.
    fn open_connection(&self) -> SqlResult<Connection> {
        let conn = Connection::open(&self.db_path)?;
        conn.pragma_update(None, "key", Self::DEFAULT_KEY)?;
        Ok(conn)
    }

    /// Initializes the database schema.
    pub fn init_db(&self) -> AuthResult<()> {
        let conn = self
            .open_connection()
            .map_err(|e| AuthError::DatabaseError(e.to_string()))?;
        conn.execute(
            "CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                credential_json TEXT NOT NULL
            )",
            [],
        )
        .map_err(|e| AuthError::DatabaseError(e.to_string()))?;
        Ok(())
    }

    /// Saves a user's credential to the database.
    pub fn save_credential(&self, username: &str, credential: &StoredCredential) -> AuthResult<()> {
        let conn = self
            .open_connection()
            .map_err(|e| AuthError::DatabaseError(e.to_string()))?;
        let json = serde_json::to_string(credential)
            .map_err(|e| AuthError::SerializationError(e.to_string()))?;

        conn.execute(
            "INSERT INTO users (username, credential_json) VALUES (?1, ?2)",
            params![username, json],
        )
        .map_err(|e| AuthError::DatabaseError(e.to_string()))?;
        Ok(())
    }

    /// Retrieves a user's credential from the database.
    pub fn get_credential(&self, username: &str) -> AuthResult<Option<StoredCredential>> {
        let conn = self
            .open_connection()
            .map_err(|e| AuthError::DatabaseError(e.to_string()))?;
        let mut stmt = conn
            .prepare("SELECT credential_json FROM users WHERE username = ?1")
            .map_err(|e| AuthError::DatabaseError(e.to_string()))?;

        let mut rows = stmt
            .query_map(params![username], |row| {
                let json: String = row.get(0)?;
                Ok(json)
            })
            .map_err(|e| AuthError::DatabaseError(e.to_string()))?;

        if let Some(json_res) = rows.next() {
            let json = json_res.map_err(|e| AuthError::DatabaseError(e.to_string()))?;
            let credential: StoredCredential = serde_json::from_str(&json)
                .map_err(|e| AuthError::SerializationError(e.to_string()))?;
            return Ok(Some(credential));
        }
        Ok(None)
    }

    /// Deletes a user's credential from the database.
    pub fn delete_credential(&self, username: &str) -> AuthResult<()> {
        let conn = self
            .open_connection()
            .map_err(|e| AuthError::DatabaseError(e.to_string()))?;
        conn.execute("DELETE FROM users WHERE username = ?1", params![username])
            .map_err(|e| AuthError::DatabaseError(e.to_string()))?;
        Ok(())
    }

    /// Checks if a user exists in the database.
    pub fn user_exists(&self, username: &str) -> AuthResult<bool> {
        let conn = self
            .open_connection()
            .map_err(|e| AuthError::DatabaseError(e.to_string()))?;
        let mut stmt = conn
            .prepare("SELECT 1 FROM users WHERE username = ?1")
            .map_err(|e| AuthError::DatabaseError(e.to_string()))?;
        let exists = stmt
            .exists(params![username])
            .map_err(|e| AuthError::DatabaseError(e.to_string()))?;
        Ok(exists)
    }
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

    /// Create a new user account
    pub fn create_account(username: &str, password: &str) -> AuthResult<()> {
        let manager = CredentialManager::with_default_path()?;
        manager.init_db()?;

        // Check if user already exists
        if manager.user_exists(username)? {
            return Err(AuthError::UserAlreadyExists(username.to_string()));
        }

        // Hash password
        let password_hash = Self::hash_password(password)?;

        // Create credential
        let credential = StoredCredential {
            password_hash,
            created_at: chrono::Utc::now(),
        };

        // Store
        manager.save_credential(username, &credential)?;

        Ok(())
    }

    /// Verify login credentials
    pub fn login(username: &str, password: &str) -> AuthResult<bool> {
        let manager = CredentialManager::with_default_path()?;
        manager.init_db()?;

        // Get stored credential
        let credential = manager
            .get_credential(username)?
            .ok_or_else(|| AuthError::UserNotFound(username.to_string()))?;

        // Verify password
        Self::verify_password(password, &credential.password_hash)
    }

    /// Delete a user account
    pub fn delete_account(username: &str) -> AuthResult<()> {
        let manager = CredentialManager::with_default_path()?;
        manager.delete_credential(username)
    }

    /// Check if a user exists
    pub fn user_exists(username: &str) -> AuthResult<bool> {
        let manager = CredentialManager::with_default_path()?;
        manager.user_exists(username)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn setup_test_manager() -> (CredentialManager, tempfile::TempDir) {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_credentials.db");
        let manager = CredentialManager::new(db_path);
        manager.init_db().unwrap();
        (manager, dir)
    }

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
    fn test_credential_manager_crud() {
        let (manager, _dir) = setup_test_manager();
        let username = "test_user";
        let credential = StoredCredential {
            password_hash: "dummy_hash".to_string(),
            created_at: chrono::Utc::now(),
        };

        // Create
        manager.save_credential(username, &credential).unwrap();
        assert!(manager.user_exists(username).unwrap());

        // Read
        let retrieved = manager.get_credential(username).unwrap().unwrap();
        assert_eq!(retrieved.password_hash, credential.password_hash);

        // Delete
        manager.delete_credential(username).unwrap();
        assert!(!manager.user_exists(username).unwrap());
        assert!(manager.get_credential(username).unwrap().is_none());
    }
}
