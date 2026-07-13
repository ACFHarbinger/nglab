import os

# Gunicorn Configuration

# Binding
bind = "0.0.0.0:8000"

# Worker Options
# For GPU inference with asyncio (FastAPI), we use uvicorn workers.
# Since GPU is a shared resource, we don't want too many processes contending for it.
# Usually 1-2 workers per GPU is enough if using batching.
workers = int(os.getenv("GUNICORN_WORKERS", "1"))
worker_class = "uvicorn.workers.UvicornWorker"

# Timeout
# Inference might take time if batching is heavy, but usually it's fast.
timeout = 60
keepalive = 2

# Logging
loglevel = "info"
accesslog = "-"  # stdout
errorlog = "-"  # stderr

# Process Naming
proc_name = "nglab_inference"

# Daemon
daemon = False
