import psutil
from flask import Flask, jsonify
import torch

app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health():
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else None

    return jsonify(
        {
            "status": "healthy",
            "system": {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "gpu_available": gpu_available,
                "gpu_name": gpu_name,
            },
        }
    )


@app.route("/ready", methods=["GET"])
def ready():
    # Placeholder for model readiness check
    return jsonify({"ready": True})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
