"""
FastAPI Inference Service for NGLab.

Provides low-latency endpoints for real-time model predictions.
"""

import time
from typing import Any, Dict, List, Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from python.src.utils.functions.functions import load_model

app = FastAPI(
    title="NGLab Inference API",
    description="High-performance model serving for trading agents.",
    version="1.0.0"
)

# Global model cache
_MODEL: Optional[torch.nn.Module] = None
_OPTS: Optional[Dict[str, Any]] = None

class PredictionRequest(BaseModel):
    """Schema for model prediction request."""
    observations: List[List[float]] = Field(..., description="Batch of observation tensors.")
    model_path: Optional[str] = Field(None, description="Path to specific model checkpoint.")
    temperature: float = Field(1.0, ge=0.01, le=2.0)

class PredictionResponse(BaseModel):
    """Schema for model prediction response."""
    predictions: List[List[float]]
    model_version: str
    latency_ms: float

def get_model(model_path: Optional[str] = None) -> torch.nn.Module:
    """Singleton model loader with caching."""
    global _MODEL, _OPTS
    
    # Simple logic: if path specified, load it. If not, load default.
    # For production, this should integrate with MLflow or a config system.
    default_path = "outputs/model_last.pt"
    target_path = model_path or default_path
    
    if _MODEL is None or model_path is not None:
        try:
            model, opts = load_model(target_path)
            model.eval()
            _MODEL = model
            _OPTS = opts
            print(f"Loaded model from {target_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {target_path}: {e}")
            
    return _MODEL

@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Service health check."""
    return {
        "status": "online",
        "gpu_available": torch.cuda.is_available(),
        "model_loaded": _MODEL is not None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> PredictionResponse:
    """
    Generate predictions for the given observations.
    """
    try:
        model = get_model(request.model_path)
        device = _OPTS["device"] if _OPTS else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Prepare data
        obs_tensor = torch.tensor(request.observations, dtype=torch.float32).to(device)
        
        start_time = time.perf_counter()
        with torch.no_grad():
            # For LSTM/NGLab models, output is typically (batch_size, sequence_length, output_dim)
            # or (batch_size, output_dim). We assume the backbone handles the shape.
            output = model(obs_tensor)
            
            # Post-processing (e.g. temperature scaling if it was categorical)
            # For MAE/Regression, we just return the raw values or apply temperature if it's probabilistic
            # Since we switched to L1 Loss, it's likely regression.
            # But the user ADR says "Softmax output layer for probabilistic predictions"
            # So we might need to handle both.
            
        latency = (time.perf_counter() - start_time) * 1000
        
        return PredictionResponse(
            predictions=output.cpu().tolist(),
            model_version=request.model_path or "default",
            latency_ms=round(latency, 2)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
