/*!
 * Main entry point for the NGLab Tauri application.
 *
 * This binary boots the Tauri framework and delegates logic to the library crate.
 */

// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

/** Launch the NGLab Tauri application. */
fn main() {
    tauri_app_lib::run()
}
