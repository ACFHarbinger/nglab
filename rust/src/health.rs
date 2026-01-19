use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct HealthStatus {
    pub status: String,
    pub version: String,
    pub uptime_seconds: u64,
    pub components: ComponentHealth,
}

#[derive(Serialize, Deserialize)]
pub struct ComponentHealth {
    pub arena: bool,
    pub orderbook: bool,
    pub polymarket_scraper: bool,
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
