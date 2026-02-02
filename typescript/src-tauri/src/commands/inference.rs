/*!
 * Model inference commands.
 */

use crate::state::ArenaState;
use nglab::models::inference::{
    list_csv_columns as backend_list_columns, list_trained_models as backend_list_models,
    predict_trained_model as backend_predict, train_model_session, ModelMetadata, TrainingParams,
};
use tauri::{Emitter, State};

/**
 * Lists available trained machine learning models.
 */
#[tauri::command]
pub async fn list_trained_models(_app: tauri::AppHandle) -> Result<Vec<ModelMetadata>, String> {
    backend_list_models()
}

/**
 * Runs inference on a selected trained model.
 */
#[tauri::command]
pub async fn predict_trained_model(
    model_name: Option<String>,
    input_data: Vec<f64>,
    state: State<'_, ArenaState>,
) -> Result<Vec<f64>, String> {
    let target_model = if let Some(name) = model_name {
        if name.is_empty() {
            None
        } else {
            Some(name)
        }
    } else {
        None
    };

    let model_to_use = match target_model {
        Some(name) => name,
        None => {
            let guard = state.active_model.lock().unwrap();
            guard
                .as_ref()
                .ok_or("No model specified and no active model selected")?
                .clone()
        }
    };

    backend_predict(model_to_use, input_data).await
}

#[tauri::command]
pub fn set_active_model(model_name: String, state: State<'_, ArenaState>) -> Result<(), String> {
    let mut guard = state.active_model.lock().unwrap();
    *guard = Some(model_name);
    Ok(())
}

#[tauri::command]
pub fn get_active_model(state: State<'_, ArenaState>) -> Result<Option<String>, String> {
    let guard = state.active_model.lock().unwrap();
    Ok(guard.clone())
}

/**
 * Lists available columns in a CSV file.
 */
#[tauri::command]
pub async fn list_csv_columns(csv_path: String) -> Result<Vec<String>, String> {
    backend_list_columns(&csv_path)
}

/**
 * Trains a supervised learning model from CSV data.
 */
#[tauri::command]
#[allow(clippy::too_many_arguments)]
pub async fn train_model(
    csv_path: String,
    target_column: String,
    model_name: String,
    epochs: u32,
    batch_size: u32,
    learning_rate: f64,
    seq_len: u32,
    pred_len: u32,
    train_split: f64,
    model_params: serde_json::Value,
    app: tauri::AppHandle,
) -> Result<String, String> {
    let params = TrainingParams {
        csv_path,
        target_column,
        model_name,
        epochs,
        batch_size,
        learning_rate,
        seq_len,
        pred_len,
        train_split,
        model_params,
    };

    train_model_session(params, move |progress| {
        let _ = app.emit("training-progress", &progress);
    })
    .await
}

/// Extended model metadata with optional performance metrics for comparison.
#[derive(serde::Serialize, Clone)]
pub struct ModelDetails {
    pub name: String,
    pub architecture: String,
    pub size_bytes: u64,
    pub modified_ts: u64,
    pub sharpe_ratio: Option<f64>,
    pub max_drawdown: Option<f64>,
    pub win_rate: Option<f64>,
    pub avg_reward: Option<f64>,
}

/// Gets detailed model information for comparison views.
#[tauri::command]
pub async fn get_model_details(model_name: String) -> Result<ModelDetails, String> {
    // Get basic metadata
    let models = backend_list_models()?;
    let meta = models
        .into_iter()
        .find(|m| m.name == model_name)
        .ok_or_else(|| format!("Model '{}' not found", model_name))?;

    // Return basic details; performance metrics would come from evaluation logs
    Ok(ModelDetails {
        name: meta.name,
        architecture: meta.architecture,
        size_bytes: meta.size_bytes,
        modified_ts: meta.modified_ts,
        sharpe_ratio: None,
        max_drawdown: None,
        win_rate: None,
        avg_reward: None,
    })
}

/// Model documentation including training config and I/O specs.
#[derive(serde::Serialize, Clone)]
pub struct ModelDocumentation {
    pub name: String,
    pub architecture: String,
    pub size_bytes: u64,
    pub modified_ts: u64,
    pub training_config: Option<serde_json::Value>,
    pub input_spec: Option<serde_json::Value>,
    pub output_spec: Option<serde_json::Value>,
}

/// Exports a model to ONNX format.
/// Returns the path to the exported ONNX file.
#[tauri::command]
pub async fn export_model_onnx(model_name: String) -> Result<String, String> {
    // Placeholder: In production, this would call Python sidecar to convert PyTorch -> ONNX
    let onnx_path = format!("models/{}.onnx", model_name);
    
    // For now, just return the expected output path
    // Actual implementation would:
    // 1. Load the PyTorch model
    // 2. Create dummy input tensor
    // 3. torch.onnx.export(model, dummy_input, onnx_path)
    Ok(onnx_path)
}

/// Gets model documentation for the documentation panel.
#[tauri::command]
pub async fn get_model_documentation(model_name: String) -> Result<ModelDocumentation, String> {
    // Get basic metadata
    let models = backend_list_models()?;
    let meta = models
        .into_iter()
        .find(|m| m.name == model_name)
        .ok_or_else(|| format!("Model '{}' not found", model_name))?;

    // Return documentation; additional metadata would be loaded from checkpoint files
    Ok(ModelDocumentation {
        name: meta.name,
        architecture: meta.architecture.clone(),
        size_bytes: meta.size_bytes,
        modified_ts: meta.modified_ts,
        training_config: None, // Would be loaded from model checkpoint metadata
        input_spec: None,      // Would be loaded from model checkpoint metadata
        output_spec: None,     // Would be loaded from model checkpoint metadata
    })
}
