use serde::{Deserialize, Serialize};

/**
 * Overall health status of the application.
 */
#[derive(Serialize, Deserialize)]
pub struct HealthStatus {
    /// Combined status: "healthy", "degraded", or "unhealthy".
    pub status: String,
    /// Application version string.
    pub version: String,
    /// Total uptime in seconds since last restart.
    pub uptime_seconds: u64,
    /// Individual health of core components.
    pub components: ComponentHealth,
}

/**
 * Health status of individual system components.
 */
#[derive(Serialize, Deserialize)]
pub struct ComponentHealth {
    /// Simulation arena status.
    pub arena: bool,
    /// Order matching engine status.
    pub orderbook: bool,
    /// Market data scrapers status.
    pub polymarket_scraper: bool,
    /// Rust-to-Python bridge status.
    pub python_binding: bool,
}

// TODO: When ArenaState is available in src-tauri, we can enable this.
// For now, providing the logic struct.
// #[tauri::command]
// pub fn health_check(state: tauri::State<ArenaState>) -> Result<HealthStatus, String> { ... }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_struct_serialization() {
        let status = HealthStatus {
            status: "healthy".to_string(),
            version: "0.1.0".to_string(),
            uptime_seconds: 100,
            components: ComponentHealth {
                arena: true,
                orderbook: true,
                polymarket_scraper: true,
                python_binding: false,
            },
        };

        let json = serde_json::to_string(&status).unwrap();
        assert!(json.contains("healthy"));
        assert!(json.contains("uptime_seconds"));
    }
}
