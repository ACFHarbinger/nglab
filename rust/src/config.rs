use config::{Config, ConfigError, Environment, File};
use serde::Deserialize;

/**
 * Global application settings loaded from configuration files and environment variables.
 */
#[derive(Debug, Deserialize)]
pub struct Settings {
    /// Environment-specific settings (name, log level).
    pub environment: EnvironmentConfig,
    /// Rust simulation engine settings.
    pub rust: RustConfig,
    /// Python ML training and runtime settings.
    pub python: PythonConfig,
    /// UI-specific settings for the Tauri frontend.
    pub tauri: TauriConfig,
    /// Persistence layer settings.
    pub database: DatabaseConfig,
    /// External API integrations (Polymarket, etc.).
    pub api: ApiConfig,
}

/// Configuration for the running environment.
#[derive(Debug, Deserialize, Clone)]
pub struct EnvironmentConfig {
    /// Environment name (e.g., "development", "production").
    pub name: String,
    /// Logging verbosity level (e.g., "info", "debug").
    pub log_level: String,
    /// Whether debug mode is enabled.
    pub debug: bool,
}

/// Configuration for the Rust backend simulation engine.
#[derive(Debug, Deserialize, Clone)]
pub struct RustConfig {
    /// Tick rate for the arena in milliseconds.
    pub arena_tick_rate_ms: u64,
    /// Default depth for order book snapshots.
    pub max_order_book_depth: usize,
    /// Whether to enable metric collection.
    pub enable_metrics: bool,
}

/// Configuration for Python integration and ML training.
#[derive(Debug, Deserialize, Clone)]
pub struct PythonConfig {
    /// Path to the model checkpoint directory.
    pub model_checkpoint_dir: String,
    /// Weights & Biases mode (e.g., "online", "disabled").
    pub wandb_mode: String,
    /// Compute device ("cpu" or "cuda").
    pub device: String,
}

/// Configuration for the Tauri frontend window.
#[derive(Debug, Deserialize, Clone)]
pub struct TauriConfig {
    /// Initial window width.
    pub window_width: u32,
    /// Initial window height.
    pub window_height: u32,
    /// Whether to enable developer tools.
    pub enable_devtools: bool,
}

/// Database connection configuration.
#[derive(Debug, Deserialize, Clone)]
pub struct DatabaseConfig {
    /// Database connection URL.
    pub url: String,
    /// Maximum connection pool size.
    pub pool_size: u32,
}

/// Configuration for external APIs.
#[derive(Debug, Deserialize, Clone)]
pub struct ApiConfig {
    /// API key for Polymarket.
    pub polymarket_api_key: String,
    /// Rate limit requests per minute.
    pub rate_limit_per_minute: u32,
}

impl Settings {
    /// Loads configuration settings based on the specified environment.
    ///
    /// # Arguments
    /// * `env` - The environment name to load (e.g., "development", "production").
    pub fn new(env: &str) -> Result<Self, ConfigError> {
        let run_mode = std::env::var("RUN_MODE").unwrap_or_else(|_| "development".into());
        let env_name = if env.is_empty() { &run_mode } else { env };

        let builder = Config::builder()
            // Start with defaults (could be added here)
            // Load configuration file
            .add_source(File::with_name(&format!("config/{}", env_name)).required(false))
            // Also try loading from rust/config dir if running from repo root
            .add_source(File::with_name(&format!("rust/config/{}", env_name)).required(false))
            // Add environment variables (overrides)
            .add_source(Environment::with_prefix("NGLAB").separator("__"));

        builder.build()?.try_deserialize()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_development_config() {
        // We assume the test is run from rust/ directory
        // The config path in new() assumes ../config or config/
        // When running cargo test in rust/, the CWD is rust/
        // So ../config/development should work if we are in rust/
        // But let's verify CWD hypothesis or adjust path

        let settings = Settings::new("development");
        match settings {
            Ok(s) => {
                assert_eq!(s.environment.name, "development");
                assert_eq!(s.rust.arena_tick_rate_ms, 100);
            }
            Err(e) => {
                // Should not fail if files exist
                panic!("Failed to load config: {}", e);
            }
        }
    }
}
