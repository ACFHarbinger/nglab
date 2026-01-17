/*!
 * Model inference commands.
 */

/**
 * Lists available trained machine learning models in the local directory.
 *
 * Scans the `python/trained_models` directory for files with the `.pt` extension.
 *
 * @param _app Tauri application handle.
 * @return Returns a list of available model names.
 */
#[tauri::command]
pub async fn list_trained_models(_app: tauri::AppHandle) -> Result<Vec<String>, String> {
    let path = std::path::Path::new("../../model_weights");
    if !path.exists() {
        return Ok(vec![]);
    }

    let mut models = Vec::new();
    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries {
            if let Ok(entry) = entry {
                let path = entry.path();
                if path.extension().map_or(false, |ext| ext == "pt") {
                    if let Some(name) = path.file_stem().and_then(|s| s.to_str()) {
                        models.push(name.to_string());
                    }
                }
            }
        }
    }
    models.sort();
    Ok(models)
}

/**
 * Response structure for a model prediction request.
 */
#[derive(serde::Serialize)]
pub struct PredictionResponse {
    /** Status of the prediction task ("success" or "error"). */
    pub status: String,
    /** Predicted values from the model. */
    pub prediction: Option<Vec<f64>>,
    /** Optional error or informative message. */
    pub message: Option<String>,
}

/**
 * Raw output format expected from the Python inference script.
 */
#[derive(serde::Deserialize)]
pub struct InferenceOutput {
    /** Status of the Python script execution. */
    pub status: String,
    /** Raw prediction data as JSON value. */
    pub prediction: Option<serde_json::Value>,
    /** Optional error message from Python. */
    pub message: Option<String>,
}

/**
 * Runs inference on a selected trained model using the Python runtime.
 *
 * This command executes a Python process to load a `.pt` model and perform
 * inference on the provided input data.
 *
 * @param model_name Name of the model file (without extension).
 * @param input_data Input vector to pass to the model.
 * @return Returns a vector of predicted values.
 */
#[tauri::command]
pub async fn predict_trained_model(
    model_name: String,
    input_data: Vec<f64>,
) -> Result<Vec<f64>, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let model_path = format!("../../model_weights/{}.pt", model_name);

        let mut cmd = std::process::Command::new("python3");
        cmd.arg("../../python/src/infer.py")
            .arg("--model_path")
            .arg(&model_path)
            .arg("--input_json")
            .arg(serde_json::to_string(&input_data).unwrap_or("[]".to_string()));

        let output = cmd
            .output()
            .map_err(|e| format!("Failed to execute python: {}", e))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!("Inference script failed: {}", stderr));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let result: InferenceOutput = serde_json::from_str(&stdout)
            .map_err(|e| format!("Failed to parse inference output: {} | raw: {}", e, stdout))?;

        if result.status == "success" {
            // Normalize prediction to Vec<f64>
            if let Some(val) = result.prediction {
                if let Some(arr) = val.as_array() {
                    let vec: Vec<f64> = arr.iter().filter_map(|v| v.as_f64()).collect();
                    return Ok(vec);
                } else if let Some(num) = val.as_f64() {
                    return Ok(vec![num]);
                }
            }
            Err("Empty or invalid prediction format".to_string())
        } else {
            Err(result.message.unwrap_or("Unknown error".to_string()))
        }
    })
    .await
    .map_err(|e| format!("Task failed: {}", e))?
}

/**
 * Training progress message structure.
 */
#[derive(serde::Serialize, serde::Deserialize, Clone)]
pub struct TrainingProgress {
    #[serde(rename = "type")]
    pub msg_type: String,
    pub epoch: Option<u32>,
    pub total_epochs: Option<u32>,
    pub train_loss: Option<f64>,
    pub val_loss: Option<f64>,
    pub percent: Option<f64>,
    pub status: Option<String>,
    pub model_path: Option<String>,
    pub message: Option<String>,
    pub columns: Option<Vec<String>>,
}

/**
 * Lists available columns in a CSV file for training target selection.
 *
 * @param csv_path Path to the CSV file.
 * @return Returns a list of column names.
 */
#[tauri::command]
pub async fn list_csv_columns(csv_path: String) -> Result<Vec<String>, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let mut cmd = std::process::Command::new("python3");
        cmd.arg("-m")
            .arg("pipeline.lightning.supervised_learning")
            .arg("--csv_path")
            .arg(&csv_path)
            .arg("--target_column")
            .arg("dummy")
            .arg("--list_columns")
            .current_dir("../../python/src");

        let output = cmd
            .output()
            .map_err(|e| format!("Failed to execute python: {}", e))?;

        let stdout = String::from_utf8_lossy(&output.stdout);

        let result: TrainingProgress = serde_json::from_str(&stdout)
            .map_err(|e| format!("Failed to parse output: {} | raw: {}", e, stdout))?;

        result
            .columns
            .ok_or_else(|| "No columns returned".to_string())
    })
    .await
    .map_err(|e| format!("Task failed: {}", e))?
}

/**
 * Trains a supervised learning model from CSV data.
 *
 * Spawns a Python process that streams training progress via stdout.
 * Progress is emitted as Tauri events to the frontend.
 *
 * @param csv_path Path to the CSV training data.
 * @param target_column Column name to predict.
 * @param model_name Model architecture name.
 * @param epochs Number of training epochs.
 * @param batch_size Training batch size.
 * @param learning_rate Learning rate.
 * @param seq_len Input sequence length.
 * @param pred_len Prediction length.
 * @param train_split Train/validation split ratio.
 * @param model_params Additional model parameters as JSON.
 * @param app Tauri application handle for emitting events.
 * @return Returns the path to the saved model on success.
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
    use std::io::BufRead;
    use tauri::Emitter;

    let result = tauri::async_runtime::spawn_blocking(move || {
        let model_params_str = serde_json::to_string(&model_params).unwrap_or("{}".to_string());

        let mut cmd = std::process::Command::new("python3");
        cmd.arg("-m")
            .arg("pipeline.lightning.supervised_learning")
            .arg("--csv_path")
            .arg(&csv_path)
            .arg("--target_column")
            .arg(&target_column)
            .arg("--model_name")
            .arg(&model_name)
            .arg("--epochs")
            .arg(epochs.to_string())
            .arg("--batch_size")
            .arg(batch_size.to_string())
            .arg("--learning_rate")
            .arg(learning_rate.to_string())
            .arg("--seq_len")
            .arg(seq_len.to_string())
            .arg("--pred_len")
            .arg(pred_len.to_string())
            .arg("--train_split")
            .arg(train_split.to_string())
            .arg("--model_params")
            .arg(&model_params_str)
            .current_dir("../../python/src")
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());

        let mut child = cmd
            .spawn()
            .map_err(|e| format!("Failed to spawn python: {}", e))?;

        let stdout = child.stdout.take().ok_or("Failed to get stdout")?;
        let reader = std::io::BufReader::new(stdout);

        let mut final_model_path: Option<String> = None;

        for line in reader.lines() {
            if let Ok(line) = line {
                if let Ok(progress) = serde_json::from_str::<TrainingProgress>(&line) {
                    // Emit progress event
                    let _ = app.emit("training-progress", &progress);

                    // Check if training complete
                    if progress.msg_type == "complete" {
                        final_model_path = progress.model_path;
                    } else if progress.msg_type == "error" {
                        return Err(progress.message.unwrap_or("Training error".to_string()));
                    }
                }
            }
        }

        // Wait for process to finish
        let status = child
            .wait()
            .map_err(|e| format!("Failed to wait for process: {}", e))?;

        if !status.success() {
            return Err("Training process failed".to_string());
        }

        final_model_path.ok_or_else(|| "No model path returned".to_string())
    })
    .await
    .map_err(|e| format!("Task failed: {}", e))?;

    result
}
