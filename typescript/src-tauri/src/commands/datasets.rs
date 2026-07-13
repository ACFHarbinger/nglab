/*!
 * Dataset management commands.
 */

use serde::Serialize;
use std::fs;
use std::path::Path;
use std::time::UNIX_EPOCH;

#[derive(Serialize)]
pub struct DatasetInfo {
    pub name: String,
    pub path: String,
    pub size: u64,
    pub last_modified: u64,
}

/**
 * Lists all CSV datasets in the data/polymarket directory.
 */
#[tauri::command]
pub async fn list_datasets() -> Result<Vec<DatasetInfo>, String> {
    let data_dir = Path::new("data/polymarket");
    if !data_dir.exists() {
        // Try creating it if it doesn't exist to avoid errors in the UI
        let _ = fs::create_dir_all(data_dir);
        return Ok(Vec::new());
    }

    let mut datasets = Vec::new();
    if let Ok(entries) = fs::read_dir(data_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() && path.extension().map_or(false, |ext| ext == "csv") {
                let metadata = entry.metadata().map_err(|e| e.to_string())?;
                let last_modified = metadata
                    .modified()
                    .map(|t| t.duration_since(UNIX_EPOCH).unwrap_or_default().as_secs())
                    .unwrap_or(0);

                datasets.push(DatasetInfo {
                    name: entry.file_name().to_string_lossy().to_string(),
                    path: path.to_string_lossy().to_string(),
                    size: metadata.len(),
                    last_modified,
                });
            }
        }
    }

    // Sort by last modified (newest first)
    datasets.sort_by(|a, b| b.last_modified.cmp(&a.last_modified));

    Ok(datasets)
}

/**
 * Deletes a specific dataset file.
 */
#[tauri::command]
pub async fn delete_dataset(path: String) -> Result<(), String> {
    let p = Path::new(&path);

    // Safety check: ensure the path is within the data directory
    if !p.starts_with("data/polymarket") {
        return Err("Unauthorized: Access denied to system files.".to_string());
    }

    if !p.exists() {
        return Err("Dataset not found.".to_string());
    }

    fs::remove_file(p).map_err(|e| format!("Failed to delete dataset: {}", e))
}
