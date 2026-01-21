/*!
 * Vault module for secure storage using SQLCipher.
 *
 * The entire database file is encrypted using SQLCipher.
 * Encryption keys are derived using Argon2.
 */

use argon2::{password_hash::SaltString, Argon2};
use rusqlite::{params, Connection, Result as SqlResult};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

/// Summary of a vault entry (for listing)
/// Summary of a vault entry (for listing).
#[derive(Debug, Serialize, Deserialize)]
pub struct VaultSummary {
    /// Unique identifier for the entry.
    pub id: i64,
    /// Human-readable label for the secret.
    pub label: String,
    /// Creation timestamp as a string.
    pub created_at: String,
}

/// Full vault entry with sensitive value
/// Full vault entry with sensitive value.
#[derive(Debug, Serialize, Deserialize)]
pub struct VaultEntry {
    /// Unique identifier for the entry.
    pub id: i64,
    /// Human-readable label for the secret.
    pub label: String,
    /// The actual decrypted secret value.
    pub value: String,
    /// Creation timestamp as a string.
    pub created_at: String,
}

/// Standard response wrapper for vault operations
/// Standard response wrapper for vault operations.
#[derive(Debug, Serialize, Deserialize)]
pub struct VaultResponse<T> {
    /// Whether the operation was successful.
    pub success: bool,
    /// Response or error message.
    pub message: String,
    /// Generic data payload (results).
    pub data: Option<T>,
}

impl<T> VaultResponse<T> {
    /// Creates a successful vault operation response.
    pub fn success(message: &str, data: Option<T>) -> Self {
        Self {
            success: true,
            message: message.to_string(),
            data,
        }
    }

    /// Creates an error vault operation response.
    pub fn error(message: &str) -> Self {
        Self {
            success: false,
            message: message.to_string(),
            data: None,
        }
    }
}

/// Manages encrypted secrets storage using SQLCipher.
pub struct VaultManager {
    db_path: PathBuf,
}

impl VaultManager {
    /// Creates a new VaultManager with a custom database path.
    pub fn new(db_path: PathBuf) -> Self {
        Self { db_path }
    }

    /// Get the default vault path in assets/secrets.
    pub fn get_default_path() -> Result<PathBuf, String> {
        // Relocate to assets/secrets in the project root
        // Using an absolute path for now as per project requirements
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

    /// Create a manager with the default path.
    pub fn with_default_path() -> Result<Self, String> {
        Ok(Self::new(Self::get_default_path()?))
    }

    /// Open an encrypted connection to the database
    pub fn open_connection(&self, key: &str) -> SqlResult<Connection> {
        let conn = Connection::open(&self.db_path)?;
        // SQLCipher encryption key
        conn.pragma_update(None, "key", key)?;
        Ok(conn)
    }

    /// Initialize the database schema
    pub fn init_db(&self, key: &str) -> SqlResult<()> {
        let conn = self.open_connection(key)?;
        conn.execute(
            "CREATE TABLE IF NOT EXISTS secrets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                label TEXT NOT NULL,
                value TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )",
            [],
        )?;
        Ok(())
    }

    /// Derive an encryption key string from a master password using Argon2
    pub fn derive_key_string(master_password: &str, salt: &[u8]) -> String {
        let mut key = [0u8; 32];
        let argon2 = Argon2::default();
        let salt_str = SaltString::encode_b64(salt).expect("Failed to encode salt");

        let _ = argon2.hash_password_into(
            master_password.as_bytes(),
            salt_str.as_str().as_bytes(),
            &mut key,
        );

        // Return as hex-encoded string for SQLCipher
        key.iter().map(|b| format!("{:02x}", b)).collect()
    }

    /// Add a secret to the vault
    pub fn add_secret(&self, key_str: &str, label: &str, value: &str) -> SqlResult<i64> {
        let conn = self.open_connection(key_str)?;
        conn.execute(
            "INSERT INTO secrets (label, value) VALUES (?1, ?2)",
            params![label, value],
        )?;
        Ok(conn.last_insert_rowid())
    }

    /// List all secrets (summaries only)
    pub fn list_secrets(&self, key_str: &str) -> SqlResult<Vec<VaultSummary>> {
        let conn = self.open_connection(key_str)?;
        let mut stmt = conn.prepare("SELECT id, label, created_at FROM secrets")?;
        let rows = stmt.query_map([], |row| {
            Ok(VaultSummary {
                id: row.get(0)?,
                label: row.get(1)?,
                created_at: row.get(2)?,
            })
        })?;

        let mut entries = Vec::new();
        for row in rows {
            entries.push(row?);
        }
        Ok(entries)
    }

    /// Get a specific secret
    pub fn get_secret(&self, key_str: &str, id: i64) -> SqlResult<Option<VaultEntry>> {
        let conn = self.open_connection(key_str)?;
        let mut stmt =
            conn.prepare("SELECT id, label, value, created_at FROM secrets WHERE id = ?1")?;
        let mut rows = stmt.query_map(params![id], |row| {
            Ok(VaultEntry {
                id: row.get(0)?,
                label: row.get(1)?,
                value: row.get(2)?,
                created_at: row.get(3)?,
            })
        })?;

        if let Some(entry_res) = rows.next() {
            return Ok(Some(entry_res?));
        }
        Ok(None)
    }

    /// Delete a secret
    pub fn delete_secret(&self, key_str: &str, id: i64) -> SqlResult<()> {
        let conn = self.open_connection(key_str)?;
        conn.execute("DELETE FROM secrets WHERE id = ?1", params![id])?;
        Ok(())
    }
}
