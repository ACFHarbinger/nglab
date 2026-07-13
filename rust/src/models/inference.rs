use serde::{Deserialize, Serialize};
use std::io::BufRead;
use std::path::Path;
use std::process::{Command, Stdio};

/// Internal helper to get the best available python command (local venv vs system)
fn get_python_command() -> Command {
    // We expect to be running from typescript/src-tauri in dev mode
    let venv_path = "../../.venv/bin/python";
    if Path::new(venv_path).exists() {
        Command::new(venv_path)
    } else {
        Command::new("python3")
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ModelMetadata {
    pub name: String,
    pub filename: String,
    pub size_bytes: u64,
    pub modified_ts: u64,
    pub architecture: String,
}

/// Lists available trained machine learning models in the local directory with metadata.
pub fn list_trained_models() -> Result<Vec<ModelMetadata>, String> {
    let path = Path::new("../../model_weights");
    if !path.exists() {
        return Ok(vec![]);
    }

    let mut models = Vec::new();
    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().is_some_and(|ext| ext == "pt") {
                if let Some(file_name) = path.file_name().and_then(|s| s.to_str()) {
                    if let Ok(metadata) = path.metadata() {
                        let size_bytes = metadata.len();
                        let modified_ts = metadata
                            .modified()
                            .ok()
                            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                            .map(|d| d.as_secs())
                            .unwrap_or(0);

                        let name = path
                            .file_stem()
                            .and_then(|s| s.to_str())
                            .unwrap_or("unknown")
                            .to_string();

                        // Heuristic for architecture
                        let architecture = if name.contains("lstm") {
                            "LSTM".to_string()
                        } else if name.contains("transformer") {
                            "Transformer".to_string()
                        } else if name.contains("cnn") || name.contains("conv") {
                            "CNN".to_string()
                        } else if name.contains("mamba") {
                            "Mamba".to_string()
                        } else {
                            "Neural Network".to_string()
                        };

                        models.push(ModelMetadata {
                            name,
                            filename: file_name.to_string(),
                            size_bytes,
                            modified_ts,
                            architecture,
                        });
                    }
                }
            }
        }
    }
    // Sort by modified time descending (newest first)
    models.sort_by(|a, b| b.modified_ts.cmp(&a.modified_ts));
    Ok(models)
}

/// Response structure for a model prediction request.
#[derive(Serialize, Deserialize, Debug)]
pub struct PredictionResponse {
    /// Status of the request ("success" or "error").
    pub status: String,
    /// Predicted values if successful.
    pub prediction: Option<Vec<f64>>,
    /// Error message if status is "error".
    pub message: Option<String>,
}

/// Raw output format expected from the Python inference script.
#[derive(Serialize, Deserialize, Debug)]
pub struct InferenceOutput {
    /// Status of the batch inference.
    pub status: String,
    /// Raw prediction values from Python.
    pub prediction: Option<serde_json::Value>,
    /// Error details if any.
    pub message: Option<String>,
}

/// Runs inference on a selected trained model using the Python runtime.
pub async fn predict_trained_model(
    model_name: String,
    input_data: Vec<f64>,
) -> Result<Vec<f64>, String> {
    let model_path = format!("../../model_weights/{}.pt", model_name);

    let mut cmd = get_python_command();
    cmd.arg("../../python/src/infer.py")
        .arg("--model_path")
        .arg(&model_path)
        .arg("--input_json")
        .arg(serde_json::to_string(&input_data).unwrap_or_else(|_| "[]".to_string()));

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
        Err(result
            .message
            .unwrap_or_else(|| "Unknown error".to_string()))
    }
}

/// Training progress message structure.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct TrainingProgress {
    /// Type of progress message (e.g., "epoch", "complete", "error").
    #[serde(rename = "type")]
    pub msg_type: String,
    /// Current epoch number.
    pub epoch: Option<u32>,
    /// Total number of epochs defined.
    pub total_epochs: Option<u32>,
    /// Calculated training loss.
    pub train_loss: Option<f64>,
    /// Calculated validation loss.
    pub val_loss: Option<f64>,
    /// Completion percentage (0.0 to 100.0).
    pub percent: Option<f64>,
    /// Current status description.
    pub status: Option<String>,
    /// Path to the saved model file (sent on completion).
    pub model_path: Option<String>,
    /// Informational or error message.
    pub message: Option<String>,
    /// List of column names (sent on column listing).
    pub columns: Option<Vec<String>>,
}

/// Lists available columns in a CSV file for training target selection.
pub fn list_csv_columns(csv_path: &str) -> Result<Vec<String>, String> {
    let mut cmd = get_python_command();
    cmd.arg("-m")
        .arg("pipeline.lightning.supervised_learning")
        .arg("--csv_path")
        .arg(csv_path)
        .arg("--target_column")
        .arg("dummy")
        .arg("--list_columns")
        .current_dir("../../python/src");

    let output = cmd
        .output()
        .map_err(|e| format!("Failed to execute python: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("Column listing script failed: {}", stderr));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);

    let result: TrainingProgress = serde_json::from_str(&stdout)
        .map_err(|e| format!("Failed to parse output: {} | raw: {}", e, stdout))?;

    result
        .columns
        .ok_or_else(|| "No columns returned".to_string())
}

/// Parameters for training a model.
#[derive(Serialize, Deserialize, Debug)]
pub struct TrainingParams {
    /// Path to the input CSV data.
    pub csv_path: String,
    /// Name of the column to use as the target variable.
    pub target_column: String,
    /// Desired name for the resulting model.
    pub model_name: String,
    /// Number of training iterations.
    pub epochs: u32,
    /// Number of samples per training batch.
    pub batch_size: u32,
    /// Learning rate for the optimizer.
    pub learning_rate: f64,
    /// Input sequence length (lookback).
    pub seq_len: u32,
    /// Prediction horizon (lookahead).
    pub pred_len: u32,
    /// Fraction of data to use for training (e.g., 0.8).
    pub train_split: f64,
    /// Advanced model hyperparameters as a JSON object.
    pub model_params: serde_json::Value,
}

/// Trains a supervised learning model from CSV data with progress callbacks.
pub async fn train_model_session<F>(
    params: TrainingParams,
    on_progress: F,
) -> Result<String, String>
where
    F: Fn(TrainingProgress) + Send + Sync + 'static,
{
    let model_params_str =
        serde_json::to_string(&params.model_params).unwrap_or_else(|_| "{}".to_string());

    let mut cmd = get_python_command();
    cmd.arg("-m")
        .arg("pipeline.lightning.supervised_learning")
        .arg("--csv_path")
        .arg(&params.csv_path)
        .arg("--target_column")
        .arg(&params.target_column)
        .arg("--model_name")
        .arg(&params.model_name)
        .arg("--epochs")
        .arg(params.epochs.to_string())
        .arg("--batch_size")
        .arg(params.batch_size.to_string())
        .arg("--learning_rate")
        .arg(params.learning_rate.to_string())
        .arg("--seq_len")
        .arg(params.seq_len.to_string())
        .arg("--pred_len")
        .arg(params.pred_len.to_string())
        .arg("--train_split")
        .arg(params.train_split.to_string())
        .arg("--model_params")
        .arg(&model_params_str)
        .current_dir("../../python/src")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = cmd
        .spawn()
        .map_err(|e| format!("Failed to spawn python: {}", e))?;

    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| "Failed to get stdout".to_string())?;
    let reader = std::io::BufReader::new(stdout);

    let mut final_model_path: Option<String> = None;

    // Use a blocking thread since Stdout reading is blocking
    let result = tokio::task::spawn_blocking(move || {
        for line in reader.lines().map_while(Result::ok) {
            if let Ok(progress) = serde_json::from_str::<TrainingProgress>(&line) {
                on_progress(progress.clone());

                if progress.msg_type == "complete" {
                    final_model_path = progress.model_path;
                } else if progress.msg_type == "error" {
                    return Err(progress
                        .message
                        .unwrap_or_else(|| "Training error".to_string()));
                }
            }
        }

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
