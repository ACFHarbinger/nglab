use serde::{Deserialize, Serialize};
use std::sync::Mutex;
use tauri::State;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Alert {
    pub id: String,
    pub alert_type: String, // "PriceAbove", "PriceBelow"
    pub target: String,
    pub value: f64,
    pub active: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Notification {
    pub id: String,
    pub title: String,
    pub message: String,
    pub timestamp: String,
    pub priority: String, // "Low", "Medium", "High"
}

pub struct AlertState {
    pub alerts: Mutex<Vec<Alert>>,
    pub notifications: Mutex<Vec<Notification>>,
}

impl Default for AlertState {
    fn default() -> Self {
        Self {
            alerts: Mutex::new(Vec::new()),
            notifications: Mutex::new(Vec::new()),
        }
    }
}

#[tauri::command]
pub fn add_alert(
    state: State<'_, AlertState>,
    alert_type: String,
    target: String,
    value: f64,
) -> Result<Alert, String> {
    let mut alerts = state.alerts.lock().map_err(|_| "Failed to lock alerts")?;
    let alert = Alert {
        id: uuid::Uuid::new_v4().to_string(),
        alert_type,
        target,
        value,
        active: true,
    };
    alerts.push(alert.clone());
    Ok(alert)
}

#[tauri::command]
pub fn get_alerts(state: State<'_, AlertState>) -> Result<Vec<Alert>, String> {
    let alerts = state.alerts.lock().map_err(|_| "Failed to lock alerts")?;
    Ok(alerts.clone())
}

#[tauri::command]
pub fn clear_notifications(state: State<'_, AlertState>) -> Result<(), String> {
    let mut notifications = state
        .notifications
        .lock()
        .map_err(|_| "Failed to lock notifications")?;
    notifications.clear();
    Ok(())
}

#[tauri::command]
pub fn get_notifications(state: State<'_, AlertState>) -> Result<Vec<Notification>, String> {
    let notifications = state
        .notifications
        .lock()
        .map_err(|_| "Failed to lock notifications")?;
    Ok(notifications.clone())
}
