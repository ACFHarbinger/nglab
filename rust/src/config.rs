use config::{Config, ConfigError, Environment, File};
use serde::Deserialize;

/**
 * Global application settings loaded from configuration files and environment variables.
 */
#[derive(Debug, Deserialize)]
pub struct Settings {
    /** Environment-specific settings (name, log level) */
    pub environment: EnvironmentConfig,
    /** Rust simulation engine settings */
    pub rust: RustConfig,
    /** Python ML training and runtime settings */
    pub python: PythonConfig,
    /** UI-specific settings for the Tauri frontend */
    pub tauri: TauriConfig,
    /** Persistence layer settings */
    pub database: DatabaseConfig,
    /** External API integrations (Polymarket, etc.) */
    pub api: ApiConfig,
}

#[derive(Debug, Deserialize)]
pub struct EnvironmentConfig {
    pub name: String,
    pub log_level: String,
    pub debug: bool,
}

#[derive(Debug, Deserialize)]
pub struct RustConfig {
    pub arena_tick_rate_ms: u64,
    pub max_order_book_depth: usize,
    pub enable_metrics: bool,
}

#[derive(Debug, Deserialize)]
pub struct PythonConfig {
    pub model_checkpoint_dir: String,
    pub wandb_mode: String,
    pub device: String,
}

#[derive(Debug, Deserialize)]
pub struct TauriConfig {
    pub window_width: u32,
    pub window_height: u32,
    pub enable_devtools: bool,
}

#[derive(Debug, Deserialize)]
pub struct DatabaseConfig {
    pub url: String,
    pub pool_size: u32,
}

#[derive(Debug, Deserialize)]
pub struct ApiConfig {
    pub polymarket_api_key: String,
    pub rate_limit_per_minute: u32,
}

impl Settings {
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
