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

from python.src.models.time_series import TimeSeriesBackbone
from python.src.utils.io.model_versioning import ModelMetadata, load_model_with_metadata


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
        config = metadata.hyperparameters
        model_config = config.get("model", config)
        model = TimeSeriesBackbone(model_config)

        # 4. Load Weights
        model, _ = load_model_with_metadata(model, model_path, map_location="cpu")
        model.eval()

        # 5. Handle Normalization
        norm_cfg = config.get("normalization")
        if norm_cfg:
            method = norm_cfg.get("method")
            if method == "minmax":
                raw_min, raw_max = norm_cfg["min"], norm_cfg["max"]
                if raw_max - raw_min > 1e-8:
                    x = (x - raw_min) / (raw_max - raw_min)
                else:
                    x = x - raw_min
            elif method == "zscore":
                raw_mean, raw_std = norm_cfg["mean"], norm_cfg["std"]
                if raw_std > 1e-8:
                    x = (x - raw_mean) / raw_std
                else:
                    x = x - raw_mean

        # 6. Run Inference
        with torch.no_grad():
            output = model(x)

        # 7. Denormalize Output
        if norm_cfg:
            method = norm_cfg.get("method")
            if method == "minmax":
                raw_min, raw_max = norm_cfg["min"], norm_cfg["max"]
                output = output * (raw_max - raw_min) + raw_min
            elif method == "zscore":
                raw_mean, raw_std = norm_cfg["mean"], norm_cfg["std"]
                output = output * raw_std + raw_mean

        # output is likely (Batch, OutDim) or (Batch, Seq, OutDim).
        # We ensure it's a flat list for the frontend.
        if output.dim() > 1:
            result = output.view(-1).tolist()
        else:
            result = [output.item()]

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
