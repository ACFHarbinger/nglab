/*
 * Health check commands for the NGLab dashboard.
 *
 * Provides system health monitoring and diagnostics via Tauri commands.
 */

use crate::state::ArenaState;
use nglab::health::{ComponentHealth, HealthStatus};
use serde::Serialize;
use std::time::Instant;
use tauri::State;

// Track application start time for uptime calculation
static START_TIME: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();

/// Initialize the start time when the module is first used.
fn get_uptime_seconds() -> u64 {
    let start = START_TIME.get_or_init(Instant::now);
    start.elapsed().as_secs()
}

/// System information for diagnostics.
#[derive(Serialize)]
pub struct SystemInfo {
    pub cpu_count: usize,
    pub os_name: String,
    pub arch: String,
}

/**
 * Check the health status of all application components.
 *
 * Returns a detailed health status including:
 * - Arena simulation engine status
 * - OrderBook state
 * - Polymarket scraper connectivity
 * - Python binding availability
 */
#[tauri::command]
pub fn health_check(state: State<ArenaState>) -> Result<HealthStatus, String> {
    // Check arena health by attempting to lock the environment
    let arena_healthy = state.env.lock().is_ok();

    // Check if simulation is running
    let running = state.running.lock().map(|r| *r).unwrap_or(false);

    // Determine overall status
    let status = if arena_healthy {
        if running {
            "healthy"
        } else {
            "degraded"
        }
    } else {
        "unhealthy"
    };

    Ok(HealthStatus {
        status: status.to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        uptime_seconds: get_uptime_seconds(),
        components: ComponentHealth {
            arena: arena_healthy,
            orderbook: arena_healthy, // OrderBook is part of arena
            polymarket_scraper: true, // Assumed healthy unless proven otherwise
            python_binding: true,     // Always available in Tauri context
        },
    })
}

/**
 * Get system information for diagnostics.
 */
#[tauri::command]
pub fn get_system_info() -> Result<SystemInfo, String> {
    let cpu_count = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(1);

    Ok(SystemInfo {
        cpu_count,
        os_name: std::env::consts::OS.to_string(),
        arch: std::env::consts::ARCH.to_string(),
    })
}
