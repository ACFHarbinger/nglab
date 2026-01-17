/*!
 * Model inference commands.
 */

use nglab::models::inference::{
    list_csv_columns as backend_list_columns, list_trained_models as backend_list_models,
    predict_trained_model as backend_predict, train_model_session, TrainingParams,
};
use tauri::Emitter;

/**
 * Lists available trained machine learning models.
 */
#[tauri::command]
pub async fn list_trained_models(_app: tauri::AppHandle) -> Result<Vec<String>, String> {
    backend_list_models()
}

/**
 * Runs inference on a selected trained model.
 */
#[tauri::command]
pub async fn predict_trained_model(
    model_name: String,
    input_data: Vec<f64>,
) -> Result<Vec<f64>, String> {
    backend_predict(model_name, input_data).await
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
