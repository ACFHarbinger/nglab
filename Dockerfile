# Stage 1: Rust builder
FROM rust:1.83-slim as rust-builder
WORKDIR /build
RUN apt-get update && apt-get install -y pkg-config libssl-dev python3-dev python3-venv
COPY rust/ ./rust/
WORKDIR /build/rust
RUN cargo build --release

# Stage 2: Python environment
FROM python:3.11-slim as python-builder
WORKDIR /app
RUN apt-get update && apt-get install -y build-essential
COPY python/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY python/src ./src
# Copy the compiled rust library (PyO3)
COPY --from=rust-builder /build/rust/target/release/libnglab.so /usr/local/lib/
ENV LD_LIBRARY_PATH=/usr/local/lib

# Stage 3: Runtime
FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=python-builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=rust-builder /build/rust/target/release/libnglab.so /usr/local/lib/
COPY python/src ./src
COPY config/ ./config/

ENV PYTHONPATH=/app/src
ENV LD_LIBRARY_PATH=/usr/local/lib
ENV NGLAB_ENV=production

EXPOSE 8000
CMD ["python", "-m", "src.main"] 
