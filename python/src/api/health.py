"""Health and readiness probe endpoints for NGLab API."""

from typing import Any

import psutil
import torch
from flask import Flask, Response, jsonify

app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health() -> Response:
    """
    Health check endpoint.
    Returns a JSON response with system status.
    """
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else None

    # Mypy now knows jsonify returns a Response, so no cast is needed.
    response_data: dict[str, Any] = {
        "status": "healthy",
        "system": {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "gpu_available": gpu_available,
            "gpu_name": gpu_name,
        },
    }
    return jsonify(response_data)


@app.route("/ready", methods=["GET"])
def ready() -> Response:
    """
    Readiness probe for orchestration.
    """
    # Removed redundant cast here as well
    return jsonify({"ready": True})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
