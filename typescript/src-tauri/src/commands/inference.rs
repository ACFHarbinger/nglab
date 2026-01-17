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
    let path = std::path::Path::new("../../python/trained_models");
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
        let model_path = format!("../../python/trained_models/{}.pt", model_name);

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
