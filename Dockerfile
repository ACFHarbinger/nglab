# =============================================================================
# Build Stage: Rust Components
# =============================================================================
FROM rust:1.83-slim-bookworm AS rust-builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libssl-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy Rust source code
COPY rust/ ./rust/

WORKDIR /build/rust

# Build Rust components
# We removed --locked to allow Cargo to sync if the lockfile is stale.
ENV CARGO_INCREMENTAL=0
ENV RUSTFLAGS="-C target-cpu=native -C opt-level=3"
RUN cargo build --release

# =============================================================================
# Build Stage: Python Environment
# =============================================================================
FROM python:3.11-slim-bookworm AS python-builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Copy project files
# Destination directory ends with a slash to satisfy multi-file COPY requirements
COPY pyproject.toml uv.lock .python-version ./
COPY python/ ./python/
COPY rust/ ./rust/
COPY nglab/ ./nglab/

# Copy built Rust library from previous stage
# Using a wildcard match to ensure we catch the shared library regardless of exact naming
# and ensuring the target directory exists.
RUN mkdir -p ./nglab
COPY --from=rust-builder /build/rust/target/release/libnglab.* ./nglab/_nglab.so

# Sync dependencies and build the package
# We use --no-dev to keep the production image slim
RUN uv sync --no-dev

# =============================================================================
# Final Stage: Runtime
# =============================================================================
FROM python:3.11-slim-bookworm AS runtime

# Set build-time variables
ARG NGLAB_VERSION=0.1.0
ENV VERSION=$NGLAB_VERSION

# Create non-privileged user
RUN groupadd --gid 1000 nglab && \
    useradd --uid 1000 --gid nglab --shell /bin/bash --create-home nglab

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libssl3 \
    curl \
    tini \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy the virtual environment from the builder stage
COPY --from=python-builder /app/.venv /app/.venv
COPY --from=python-builder /app/python/src /app/python/src
COPY --from=python-builder /app/nglab /app/nglab

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/python/src:$PYTHONPATH"
ENV PYTHONUNBUFFERED=1

# Expose API port
EXPOSE 8000

# Switch to non-root user
USER nglab

# Use tini as init process
ENTRYPOINT ["/usr/bin/tini", "--"]

# Start the application using uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]