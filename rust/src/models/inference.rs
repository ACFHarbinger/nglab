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

/// Lists available trained machine learning models in the local directory.
pub fn list_trained_models() -> Result<Vec<String>, String> {
    let path = Path::new("../../model_weights");
    if !path.exists() {
        return Ok(vec![]);
    }

    let mut models = Vec::new();
    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().is_some_and(|ext| ext == "pt") {
                if let Some(name) = path.file_stem().and_then(|s| s.to_str()) {
                    models.push(name.to_string());
                }
            }
        }
    }
    models.sort();
    Ok(models)
}

/// Response structure for a model prediction request.
#[derive(Serialize, Deserialize, Debug)]
pub struct PredictionResponse {
    pub status: String,
    pub prediction: Option<Vec<f64>>,
    pub message: Option<String>,
}

/// Raw output format expected from the Python inference script.
#[derive(Serialize, Deserialize, Debug)]
pub struct InferenceOutput {
    pub status: String,
    pub prediction: Option<serde_json::Value>,
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
    pub csv_path: String,
    pub target_column: String,
    pub model_name: String,
    pub epochs: u32,
    pub batch_size: u32,
    pub learning_rate: f64,
    pub seq_len: u32,
    pub pred_len: u32,
    pub train_split: f64,
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
