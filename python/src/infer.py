"""
Inference script for trained models.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

# Add src to path so we can import modules
sys.path.append(str(Path(__file__).parent))

from models.time_series import TimeSeriesBackbone
from utils.model_versioning import ModelMetadata, load_model_with_metadata


def main():
    """Run inference using a trained model and input JSON."""
    parser = argparse.ArgumentParser(description="Run inference on a trained model")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to .pt checkpoint"
    )
    parser.add_argument(
        "--input_json",
        type=str,
        required=True,
        help="JSON string of input data (list of floats)",
    )
    args = parser.parse_args()

    try:
        # 1. Parse Input Data
        input_data = json.loads(args.input_json)
        if not isinstance(input_data, list):
            raise ValueError("Input JSON must be a list of numbers")

        # Convert to tensor (Batch=1, Seq, Feat=1)
        # Assuming univariate for now as per dashboard chart data
        x = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)

        # 2. Load Metadata and Model Config
        model_path = Path(args.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # specific handling: we need to instantiate the model BEFORE loading weights.
        # But we need hyperparameters to instantiate.
        # So we try to peek at the checkpoint file first without loading strict state dict?
        # Or we read the parallel .json file if it exists (which model_versioning saves).

        metadata_path = model_path.with_suffix(".json")
        if not metadata_path.exists():
            # Fallback: try to load checkpoint with torch.load just to get metadata part
            checkpoint = torch.load(model_path, map_location="cpu")
            if "metadata" in checkpoint:
                metadata = ModelMetadata.from_dict(checkpoint["metadata"])
            else:
                raise ValueError(
                    "Checkpoint missing metadata and no sidecar JSON found."
                )
        else:
            with open(metadata_path) as f:
                metadata = ModelMetadata.from_json(f.read())

        # 3. Instantiate Model
        # metadata.hyperparameters should match the config structure expected by TimeSeriesBackbone
        # 'model' key in config usually holds the backbone params

        # NOTE: The Metadata usually stores the FULL config (cfg).
        # So it might look like { model: { ... }, task: ... }
        # Or it might be just the model config.
        # Let's assume metadata.hyperparameters is the full config used in training.

        config = metadata.hyperparameters
        model_config = config.get(
            "model", config
        )  # Fallback if it was just model config

        model = TimeSeriesBackbone(model_config)

        # 4. Load Weights
        # We explicitly rely on our util, which handles the "metadata" key inside the pt file
        # We wrap model in a way that load_model_with_metadata expects?
        # Actually load_model_with_metadata takes (model, path).

        model, _ = load_model_with_metadata(model, model_path, map_location="cpu")
        model.eval()

        # 5. Run Inference
        with torch.no_grad():
            output = model(x)

        # output is likely (Batch, Seq, OutDim) or (Batch, OutDim).
        # For time series forecasting, we probably want the last value or the full sequence?
        # If the model is sequence-to-sequence, it returns (B, L, D).
        # We'll flatten the output to a list.

        result = output.squeeze().tolist()

        # Wrap in standard response
        response = {
            "status": "success",
            "prediction": result,
            "metadata": metadata.to_dict(),
        }
        print(json.dumps(response))

    except Exception as e:
        error_response = {"status": "error", "message": str(e)}
        print(json.dumps(error_response))
        sys.exit(1)


if __name__ == "__main__":
    main()
