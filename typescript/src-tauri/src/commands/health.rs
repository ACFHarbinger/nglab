/*!
 * Health check and debug commands.
 */

use crate::state::ArenaState;
use nglab::utils::memory::{get_memory_usage as fetch_mem, MemoryStats};
use std::sync::atomic::Ordering;
use tauri::State;

/**
 * Basic health check to ensure the backend is responsive.
 */
#[tauri::command]
pub fn health_check() -> &'static str {
    "OK"
}

/**
 * Returns basic system info (placeholder).
 */
#[tauri::command]
pub fn get_system_info() -> String {
    format!("NGLab Backend - OS: {}", std::env::consts::OS)
}

/**
 * Toggles the global debug mode.
 */
#[tauri::command]
pub fn set_debug_mode(state: State<ArenaState>, enabled: bool) {
    state.debug_mode.store(enabled, Ordering::Relaxed);
    println!("Debug mode set to: {}", enabled);
}

/**
 * Returns real-time memory usage of the process.
 */
#[tauri::command]
pub fn get_memory_usage() -> Option<MemoryStats> {
    fetch_mem()
}
