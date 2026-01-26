/*!
 * Tauri commands for Paper Trading management.
 */

use crate::state::PaperState;
use nglab::simulation::paper_trading::PaperAccount;
use std::sync::atomic::Ordering;
use tauri::State;

/**
 * Get the current paper trading account state.
 */
#[tauri::command]
pub async fn get_paper_account(state: State<'_, PaperState>) -> Result<PaperAccount, String> {
    let account = state.account.lock().map_err(|e| e.to_string())?;
    Ok(account.clone())
}

/**
 * Reset the paper trading account to a fresh state.
 */
#[tauri::command]
pub async fn reset_paper_account(
    state: State<'_, PaperState>,
    initial_balance: f64,
) -> Result<(), String> {
    let mut account = state.account.lock().map_err(|e| e.to_string())?;
    account.reset(initial_balance);
    let _ = account.save("paper_account.json");
    Ok(())
}

/**
 * Toggle whether Paper Trading mode is active in the UI.
 */
#[tauri::command]
pub async fn toggle_paper_mode(state: State<'_, PaperState>, active: bool) -> Result<(), String> {
    state.active.store(active, Ordering::SeqCst);
    let account = state.account.lock().map_err(|e| e.to_string())?;
    let _ = account.save("paper_account.json");
    Ok(())
}

/**
 * Check if Paper Trading mode is currently active.
 */
#[tauri::command]
pub async fn is_paper_mode_active(state: State<'_, PaperState>) -> Result<bool, String> {
    Ok(state.active.load(Ordering::SeqCst))
}
