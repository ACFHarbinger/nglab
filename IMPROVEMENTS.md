# NGLab Codebase Improvements & Roadmap

> **Version:** 1.0
> **Date:** 2026-01-16
> **Status:** Draft
> **Purpose:** Comprehensive analysis and actionable improvements for the NGLab trading bot platform

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Current State Assessment](#current-state-assessment)
3. [Critical Improvements (P0)](#critical-improvements-p0)
4. [High Priority Improvements (P1)](#high-priority-improvements-p1)
5. [Medium Priority Improvements (P2)](#medium-priority-improvements-p2)
6. [Long-term Enhancements (P3)](#long-term-enhancements-p3)
7. [Technical Debt](#technical-debt)
8. [Performance Optimization Opportunities](#performance-optimization-opportunities)
9. [Security Hardening](#security-hardening)
10. [Documentation Improvements](#documentation-improvements)
11. [Implementation Roadmap](#implementation-roadmap)

---

## Executive Summary

NGLab is a sophisticated multimodal deep reinforcement learning platform for financial trading with a well-designed three-tier architecture (Rust simulation engine, Python ML training, TypeScript/Tauri UI). The project demonstrates strong architectural foundations and professional development practices, but requires focused improvements in testing coverage, production readiness, and operational observability before large-scale deployment.

### Project Maturity: **Early Production (α/β)**

**Strengths:**
- ✅ Clean separation of concerns across language boundaries
- ✅ Strong type safety (Rust + TypeScript + Python type hints)
- ✅ Performance-conscious design (zero-copy transfers, async I/O, benchmarking)
- ✅ Comprehensive ML infrastructure (VAE, GAN, Diffusion, RNNs, Transformers)
- ✅ CI/CD pipeline with pre-commit hooks

**Critical Gaps:**
- ❌ No frontend testing infrastructure
- ❌ Limited production deployment documentation
- ❌ Minimal logging and observability
- ❌ Incomplete error handling standardization
- ❌ No containerization support

---

## Current State Assessment

### Code Quality Metrics

| Component | LOC | Test Coverage | Documentation | CI/CD | Status |
|-----------|-----|---------------|---------------|-------|--------|
| Rust Core | ~2,000 | Unit tests + benchmarks | Good (rustdoc) | ✅ | Production-ready |
| Python ML | ~1,200 | 9 test files (~30 tests) | Moderate | ✅ | Functional |
| TypeScript UI | ~250 | None | JSDoc comments | ✅ Build only | Functional |
| Integration | N/A | Minimal E2E | Architecture.md | Partial | Needs work |

### Architecture Health: **7/10**

**What's Working:**
- Clean interfaces between Rust/Python/TypeScript layers
- PyO3 bindings are efficient and well-structured
- Tauri IPC event system is reliable
- Gymnasium-compatible environment design

**What Needs Attention:**
- Error propagation across language boundaries
- Logging consistency (different strategies per layer)
- Configuration management for production environments
- Health check and monitoring endpoints

---

## Critical Improvements (P0)

> **Timeline:** 2-4 weeks
> **Impact:** Blocks production deployment

### 1. Frontend Testing Infrastructure (Completed)
**Status:** ✅ Implemented with Vitest and Cypress (see Changelog 2026-01-18).

**Current State:** Zero automated tests for React/Tauri frontend

**Required Actions:**

#### 1.1 Setup Jest/Vitest Testing Framework
```bash
# Install dependencies
cd typescript
npm install -D vitest @vitest/ui @testing-library/react @testing-library/jest-dom happy-dom
```

**Configuration:** [typescript/vitest.config.ts](typescript/vitest.config.ts)
```typescript
import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  plugins: [react()],
  test: {
    environment: 'happy-dom',
    globals: true,
    setupFiles: ['./src/__tests__/setup.ts'],
    coverage: {
      provider: 'v8',
      reporter: ['text', 'html', 'lcov'],
      exclude: ['src-tauri/**', 'node_modules/**'],
    },
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
});
```

#### 1.2 Unit Tests for Critical Hooks

**Priority Test Files:**
- `src/hooks/__tests__/useArena.test.tsx` - Arena state management
- `src/hooks/__tests__/usePolymarket.test.tsx` - Market data streaming
- `src/components/__tests__/PriceChart.test.tsx` - Chart rendering
- `src/components/__tests__/OrderBook.test.tsx` - Order book display

**Example Test Structure:**
```typescript
// src/hooks/__tests__/useArena.test.tsx
import { renderHook, act } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { useArena } from '../useArena';
import { mockIPC } from '@tauri-apps/api/mocks';

describe('useArena', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('should initialize with default state', () => {
    const { result } = renderHook(() => useArena());
    expect(result.current.isRunning).toBe(false);
    expect(result.current.priceHistory).toEqual([]);
  });

  it('should handle arena-update events', async () => {
    const { result } = renderHook(() => useArena());

    await act(async () => {
      // Simulate Tauri event emission
      // Test implementation here
    });

    expect(result.current.priceHistory.length).toBeGreaterThan(0);
  });
});
```

#### 1.3 E2E Testing with Playwright

**Setup:**
```bash
npm install -D @playwright/test
npx playwright install chromium
```

**Critical E2E Scenarios:**
- Launch Tauri app and verify UI loads
- Start arena simulation and verify chart updates
- Interact with Polymarket scraper
- Test order book visualization

**Target Coverage:** 80% line coverage, 100% critical path coverage

**Estimated Effort:** 3-5 days

---

### 2. Production Error Handling & Logging (Completed)
**Status:** ✅ Implemented with `thiserror` and `tracing` (see Changelog 2026-01-19).

**Current State:** Standardized error handling and structured logging implemented.

#### 2.1 Standardize Rust Error Types

**Create:** [rust/src/errors.rs](rust/src/errors.rs)
```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum NGLabError {
    #[error("Order book error: {0}")]
    OrderBook(String),

    #[error("Simulation error: {0}")]
    Simulation(String),

    #[error("Market data error: {0}")]
    MarketData(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Python error: {0}")]
    Python(String),
}

impl From<NGLabError> for PyErr {
    fn from(err: NGLabError) -> PyErr {
        PyRuntimeError::new_err(err.to_string())
    }
}
```

**Action Items:**
- Replace all `unwrap()` calls with proper error handling
- Propagate errors using `Result<T, NGLabError>`
- Add context to errors using `.context()` or `.with_context()`

#### 2.2 Implement Structured Logging

**Add Dependencies:** [rust/Cargo.toml](rust/Cargo.toml)
```toml
[dependencies]
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter", "json"] }
tracing-appender = "0.2"
```

**Configuration:** [rust/src/logging.rs](rust/src/logging.rs)
```rust
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

pub fn init_logging(log_level: &str) -> Result<(), Box<dyn std::error::Error>> {
    let file_appender = tracing_appender::rolling::daily("./logs", "nglab.log");
    let (non_blocking, _guard) = tracing_appender::non_blocking(file_appender);

    tracing_subscriber::registry()
        .with(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(log_level)))
        .with(tracing_subscriber::fmt::layer())
        .with(tracing_subscriber::fmt::layer().json().with_writer(non_blocking))
        .init();

    Ok(())
}
```

**Python Logging:** [python/src/utils/logging.py](python/src/utils/logging.py)
```python
import logging
import sys
from pathlib import Path

def setup_logging(level: str = "INFO", log_file: Path | None = None):
    """Configure structured logging for Python components."""
    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
```

**Estimated Effort:** 4-6 days

---

### 3. Production Configuration Management (Completed)
**Status:** ✅ Implemented with `config` crate (see Changelog 2026-01-19).

**Current State:** Environment-specific TOML configurations implemented.

#### 3.1 Environment-Based Configuration

**Create:** [config/](config/)
```
config/
├── development.toml
├── staging.toml
├── production.toml
└── schema.json
```

**Example:** [config/production.toml](config/production.toml)
```toml
[environment]
name = "production"
log_level = "info"
debug = false

[rust]
arena_tick_rate_ms = 100
max_order_book_depth = 1000
enable_metrics = true

[python]
model_checkpoint_dir = "/var/lib/nglab/models"
wandb_mode = "online"
device = "cuda"

[tauri]
window_width = 1280
window_height = 720
enable_devtools = false

[database]
url = "${DATABASE_URL}"  # Environment variable
pool_size = 10

[api]
polymarket_api_key = "${POLYMARKET_API_KEY}"
rate_limit_per_minute = 60
```

**Configuration Loading:** [rust/src/config.rs](rust/src/config.rs)
```rust
use config::{Config, Environment, File};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct Settings {
    pub environment: EnvironmentConfig,
    pub rust: RustConfig,
    pub python: PythonConfig,
    pub tauri: TauriConfig,
    pub database: DatabaseConfig,
    pub api: ApiConfig,
}

impl Settings {
    pub fn new(env: &str) -> Result<Self, config::ConfigError> {
        Config::builder()
            .add_source(File::with_name(&format!("config/{}", env)))
            .add_source(Environment::with_prefix("NGLAB").separator("__"))
            .build()?
            .try_deserialize()
    }
}
```

**Estimated Effort:** 2-3 days

---

### 4. Health Checks & Monitoring Endpoints (Completed)

**Current State:** Completed Python API and Rust Tauri health checks.

#### 4.1 Add Health Check Endpoints

**Rust/Tauri:** [typescript/src-tauri/src/health.rs](typescript/src-tauri/src/health.rs)
```rust
use serde::{Deserialize, Serialize};
use std::time::SystemTime;

#[derive(Serialize, Deserialize)]
pub struct HealthStatus {
    pub status: String,
    pub version: String,
    pub uptime_seconds: u64,
    pub components: ComponentHealth,
}

#[derive(Serialize, Deserialize)]
pub struct ComponentHealth {
    pub arena: bool,
    pub orderbook: bool,
    pub polymarket_scraper: bool,
    pub python_binding: bool,
}

#[tauri::command]
pub fn health_check(state: tauri::State<ArenaState>) -> Result<HealthStatus, String> {
    let arena_locked = state.env.try_lock().is_ok();

    Ok(HealthStatus {
        status: if arena_locked { "healthy" } else { "degraded" },
        version: env!("CARGO_PKG_VERSION").to_string(),
        uptime_seconds: get_uptime(),
        components: ComponentHealth {
            arena: arena_locked,
            orderbook: true, // Add actual check
            polymarket_scraper: true, // Add actual check
            python_binding: test_python_binding(),
        },
    })
}
```

**Python Flask Endpoint (Optional):** [python/src/api/health.py](python/src/api/health.py)
```python
from flask import Flask, jsonify
import psutil
import torch

app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "gpu_available": torch.cuda.is_available(),
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
    })

@app.route('/ready', methods=['GET'])
def ready():
    # Check if models are loaded, etc.
    return jsonify({"ready": True})
```

**Estimated Effort:** 1-2 days

---

## High Priority Improvements (P1)

> **Timeline:** 4-8 weeks
> **Impact:** Significantly improves reliability and maintainability

### 5. Comprehensive API Documentation (Completed)
**Status:** ✅ Implemented with rustdoc, Sphinx, and TypeDoc (see Changelog 2026-01-19).

**Current State:** Fully documented Rust, Python, and TypeScript APIs.

**Action Items:**
- Add module-level documentation to all public modules
- Document all public functions with examples
- Generate docs with `cargo doc --no-deps --open`

**Example:** [rust/src/simulation/gym.rs](rust/src/simulation/gym.rs:1-50)
```rust
//! Gymnasium-compatible reinforcement learning environment for trading simulation.
//!
//! # Overview
//!
//! The `TradingEnv` provides a standard RL interface for training trading agents.
//! It supports both discrete and continuous action spaces with configurable
//! observation and reward structures.
//!
//! # Examples
//!
//! ```rust
//! use nglab::simulation::TradingEnv;
//!
//! let env = TradingEnv::new(initial_cash, max_position, tick_size);
//! let obs = env.reset(None);
//! let (next_obs, reward, done, info) = env.step(action);
//! ```
//!
//! # Performance
//!
//! - Step time: <1ms
//! - Memory usage: ~50KB base + order history
//! - Zero-copy NumPy integration via PyO3

/// Trading environment implementing the Gymnasium interface.
///
/// # Fields
///
/// * `orderbook` - Central limit order book for trade execution
/// * `position` - Current asset position (can be negative for shorts)
/// * `cash` - Available cash balance
/// * `portfolio_value` - Total portfolio value (cash + position * price)
#[pyclass]
pub struct TradingEnv {
    // ... fields
}
```

**Add cargo-doc-check to CI:**
```yaml
# .github/workflows/ci.yml
- name: Check documentation
  run: cargo doc --no-deps --all-features
  env:
    RUSTDOCFLAGS: "-D warnings"
```

#### 5.2 Python API Reference with Sphinx

**Setup:**
```bash
cd python
pip install sphinx sphinx-rtd-theme sphinx-autodoc-typehints
sphinx-quickstart docs
```

**Configuration:** [python/docs/conf.py](python/docs/conf.py)
```python
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_autodoc_typehints',
]

napoleon_google_docstring = True
napoleon_include_init_with_doc = True
```

**Generate:**
```bash
sphinx-apidoc -o docs/source python/src
sphinx-build -b html docs/source docs/build
```

#### 5.3 TypeScript API Documentation

**Add TSDoc comments and generate with typedoc:**
```bash
npm install -D typedoc
npx typedoc --out docs/typescript src/
```

**Estimated Effort:** 5-7 days

---

### 6. Containerization & Deployment (Completed)
**Status:** ✅ Implemented Docker and Docker Compose setup (see Changelog 2026-01-19).

**Current State:** Production-ready multi-stage Docker build available.

#### 6.1 Multi-Stage Docker Build

**Create:** [Dockerfile](Dockerfile)
```dockerfile
# Stage 1: Rust builder
FROM rust:1.83-slim as rust-builder
WORKDIR /build
RUN apt-get update && apt-get install -y pkg-config libssl-dev
COPY rust/ ./rust/
WORKDIR /build/rust
RUN cargo build --release

# Stage 2: Python environment
FROM python:3.11-slim as python-builder
WORKDIR /build
COPY python/ ./python/
COPY --from=rust-builder /build/rust/target/release/libnglab.so /usr/local/lib/
RUN pip install --no-cache-dir -r python/requirements.txt

# Stage 3: Runtime
FROM python:3.11-slim
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=rust-builder /build/rust/target/release/libnglab.so /usr/local/lib/
COPY --from=python-builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY python/src /app/src
COPY config/ /app/config/

WORKDIR /app
ENV PYTHONPATH=/app/src
ENV LD_LIBRARY_PATH=/usr/local/lib

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8000/health')"

CMD ["python", "-m", "src.api.server"]
```

#### 6.2 Docker Compose for Development

**Create:** [docker-compose.yml](docker-compose.yml)
```yaml
version: '3.8'

services:
  nglab-api:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - NGLAB_ENV=development
      - DATABASE_URL=postgresql://postgres:password@db:5432/nglab
    volumes:
      - ./logs:/app/logs
      - ./models:/app/models
    depends_on:
      - db
    restart: unless-stopped

  db:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: nglab
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    ports:
      - "9090:9090"
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
    depends_on:
      - prometheus

volumes:
  postgres_data:
  prometheus_data:
  grafana_data:
```

**Estimated Effort:** 3-4 days

---

### 7. Performance Profiling & Optimization (Completed)
**Status:** ✅ Implemented continuous benchmarking and Python profiling (see Changelog 2026-01-19).

**Current State:** Benchmarks monitored in CI; profiling integrated into training pipeline.

#### 7.1 Continuous Benchmarking

**Add to CI:** [.github/workflows/benchmark.yml](.github/workflows/benchmark.yml)
```yaml
name: Continuous Benchmarking

on:
  push:
    branches: [main]
  pull_request:

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable

      - name: Run benchmarks
        run: |
          cd rust
          cargo bench -- --save-baseline current

      - name: Compare with main
        if: github.event_name == 'pull_request'
        run: |
          git fetch origin main
          git checkout origin/main
          cargo bench -- --save-baseline main
          git checkout -
          cargo bench -- --baseline main

      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: benchmark-results
          path: rust/target/criterion/
```

#### 7.2 Python Profiling Integration

**Add:** [python/src/utils/profiling.py](python/src/utils/profiling.py)
```python
import cProfile
import pstats
from functools import wraps
from pathlib import Path

def profile(output_dir: Path = Path("./profiles")):
    """Decorator to profile function execution."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            profiler = cProfile.Profile()
            profiler.enable()
            result = func(*args, **kwargs)
            profiler.disable()

            output_dir.mkdir(exist_ok=True)
            stats_file = output_dir / f"{func.__name__}.prof"
            profiler.dump_stats(stats_file)

            # Print summary
            stats = pstats.Stats(profiler)
            stats.sort_stats('cumulative')
            stats.print_stats(20)

            return result
        return wrapper
    return decorator
```

**Usage:**
```python
from utils.profiling import profile

@profile()
def train_model(config):
    # Training code
    pass
```

**Estimated Effort:** 2-3 days

---

### 8. CI/CD Deployment Pipeline

**Current State:** CI builds but no automated deployment

#### 8.1 GitHub Actions Deployment Workflow

**Create:** [.github/workflows/deploy.yml](.github/workflows/deploy.yml)
```yaml
name: Deploy

on:
  push:
    tags:
      - 'v*'

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Login to Container Registry
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ghcr.io/${{ github.repository }}
          tags: |
            type=semver,pattern={{version}}
            type=semver,pattern={{major}}.{{minor}}
            type=sha

      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

      - name: Deploy to staging
        if: github.ref == 'refs/heads/main'
        run: |
          # Add deployment script here
          echo "Deploying to staging..."

      - name: Deploy to production
        if: startsWith(github.ref, 'refs/tags/v')
        run: |
          # Add production deployment script
          echo "Deploying to production..."
```

**Estimated Effort:** 2-3 days

---

### Medium Priority Improvements (P2) (Completed)
**Status:** ✅ All phases implemented (2026-01-19).

> **Timeline:** 2-3 months
> **Impact:** Enhances developer experience and code maintainability

### 9. Type Safety Improvements

#### 9.1 Python Type Stubs for Rust Bindings (Completed)
**Status:** ✅ Implemented in `nglab/_nglab.pyi` (2026-01-19).

**Generate stubs for the `nglab` Rust module:**

**Create:** [python/src/nglab.pyi](python/src/nglab.pyi)
```python
"""Type stubs for the nglab Rust module."""

from typing import Any, Literal, TypedDict
import numpy as np
import numpy.typing as npt

class TradingEnv:
    """Gymnasium-compatible trading environment."""

    def __init__(
        self,
        initial_cash: float,
        max_position: int,
        tick_size: float,
    ) -> None: ...

    def reset(self, seed: int | None = None) -> npt.NDArray[np.float64]: ...

    def step(
        self, action: int | npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], float, bool, dict[str, Any]]: ...

    def render(self, mode: Literal["human", "rgb_array"] = "human") -> None | npt.NDArray[np.uint8]: ...

    @property
    def action_space(self) -> Any: ...

    @property
    def observation_space(self) -> Any: ...

class OrderBook:
    """Central limit order book."""

    def __init__(self, tick_size: float) -> None: ...

    def add_limit_order(
        self, side: Literal["buy", "sell"], price: float, quantity: int
    ) -> str: ...

    def add_market_order(
        self, side: Literal["buy", "sell"], quantity: int
    ) -> list[Trade]: ...

    def cancel_order(self, order_id: str) -> bool: ...

    def get_best_bid(self) -> float | None: ...

    def get_best_ask(self) -> float | None: ...

    def get_mid_price(self) -> float | None: ...

class Trade(TypedDict):
    price: float
    quantity: int
    timestamp: int
    buyer_id: str
    seller_id: str

class PolymarketArena:
    """Polymarket prediction market simulation."""

    def __init__(self, initial_liquidity: float) -> None: ...

    def buy_yes_shares(self, amount: float) -> float: ...

    def buy_no_shares(self, amount: float) -> float: ...

    def get_yes_price(self) -> float: ...

    def get_no_price(self) -> float: ...
```

#### 9.2 Mypy Strict Mode (Completed)
**Status:** ✅ Enabled in `pyproject.toml` (2026-01-19).

**Update:** [python/pyproject.toml](python/pyproject.toml)
```toml
[tool.mypy]
python_version = "3.11"
strict = true
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_any_generics = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true

[[tool.mypy.overrides]]
module = [
    "nglab.*",  # Rust bindings
    "torch.*",
    "tensordict.*",
]
ignore_missing_imports = true
```

**Estimated Effort:** 3-4 days

---

### 10. Database Integration for Trade History

**Current State:** In-memory only, no persistence

#### 10.1 PostgreSQL Schema (Completed)
**Status:** ✅ Initial schema in `migrations/001_initial_schema.sql`.

**Create:** [migrations/001_initial_schema.sql](migrations/001_initial_schema.sql)
```sql
-- Trade history
CREATE TABLE trades (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(4) NOT NULL CHECK (side IN ('buy', 'sell')),
    price NUMERIC(20, 8) NOT NULL,
    quantity NUMERIC(20, 8) NOT NULL,
    value NUMERIC(20, 8) NOT NULL,
    order_id UUID NOT NULL,
    agent_id VARCHAR(50),
    metadata JSONB
);

CREATE INDEX idx_trades_timestamp ON trades(timestamp DESC);
CREATE INDEX idx_trades_symbol ON trades(symbol);
CREATE INDEX idx_trades_agent_id ON trades(agent_id);

-- Portfolio snapshots
CREATE TABLE portfolio_snapshots (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    agent_id VARCHAR(50) NOT NULL,
    cash NUMERIC(20, 8) NOT NULL,
    position NUMERIC(20, 8) NOT NULL,
    portfolio_value NUMERIC(20, 8) NOT NULL,
    sharpe_ratio NUMERIC(10, 4),
    max_drawdown NUMERIC(10, 4),
    total_return NUMERIC(10, 4)
);

CREATE INDEX idx_portfolio_timestamp ON portfolio_snapshots(timestamp DESC);
CREATE INDEX idx_portfolio_agent_id ON portfolio_snapshots(agent_id);

-- Model checkpoints
CREATE TABLE model_checkpoints (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    model_name VARCHAR(100) NOT NULL,
    version VARCHAR(50) NOT NULL,
    architecture TEXT NOT NULL,
    hyperparameters JSONB NOT NULL,
    metrics JSONB NOT NULL,
    checkpoint_path TEXT NOT NULL,
    git_commit VARCHAR(40),
    UNIQUE(model_name, version)
);

-- Market data
CREATE TABLE market_data (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    source VARCHAR(50) NOT NULL,
    bid NUMERIC(20, 8),
    ask NUMERIC(20, 8),
    last NUMERIC(20, 8),
    volume NUMERIC(20, 8),
    metadata JSONB
);

CREATE INDEX idx_market_data_symbol_timestamp ON market_data(symbol, timestamp DESC);

-- Hypertables for TimescaleDB (optional)
-- SELECT create_hypertable('trades', 'timestamp');
-- SELECT create_hypertable('portfolio_snapshots', 'timestamp');
-- SELECT create_hypertable('market_data', 'timestamp');
```

#### 10.2 SQLAlchemy Models (Completed)
**Status:** ✅ Implemented in `python/src/db/models.py`.

**Create:** [python/src/db/models.py](python/src/db/models.py)
```python
from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Numeric, DateTime, JSON, Index, CheckConstraint
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Trade(Base):
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    symbol = Column(String(20), nullable=False)
    side = Column(String(4), nullable=False)
    price = Column(Numeric(20, 8), nullable=False)
    quantity = Column(Numeric(20, 8), nullable=False)
    value = Column(Numeric(20, 8), nullable=False)
    order_id = Column(UUID(as_uuid=True), nullable=False)
    agent_id = Column(String(50))
    metadata = Column(JSONB)

    __table_args__ = (
        CheckConstraint("side IN ('buy', 'sell')", name="check_side"),
        Index("idx_trades_timestamp", "timestamp", postgresql_using="btree"),
        Index("idx_trades_symbol", "symbol"),
    )

class PortfolioSnapshot(Base):
    __tablename__ = "portfolio_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    agent_id = Column(String(50), nullable=False)
    cash = Column(Numeric(20, 8), nullable=False)
    position = Column(Numeric(20, 8), nullable=False)
    portfolio_value = Column(Numeric(20, 8), nullable=False)
    sharpe_ratio = Column(Numeric(10, 4))
    max_drawdown = Column(Numeric(10, 4))
    total_return = Column(Numeric(10, 4))
```

**Estimated Effort:** 4-5 days

---

### 11. Metrics & Observability

**Current State:** No metrics collection

#### 11.1 Prometheus Metrics Exporter

**Rust:** [rust/src/metrics.rs](rust/src/metrics.rs)
```rust
use prometheus::{
    Counter, Histogram, HistogramOpts, IntGauge, Registry, TextEncoder,
};
use std::sync::Arc;

pub struct Metrics {
    pub orders_total: Counter,
    pub trades_total: Counter,
    pub step_duration: Histogram,
    pub portfolio_value: IntGauge,
    pub orderbook_depth: IntGauge,
}

impl Metrics {
    pub fn new(registry: &Registry) -> Result<Self, Box<dyn std::error::Error>> {
        let orders_total = Counter::new("nglab_orders_total", "Total orders submitted")?;
        registry.register(Box::new(orders_total.clone()))?;

        let trades_total = Counter::new("nglab_trades_total", "Total trades executed")?;
        registry.register(Box::new(trades_total.clone()))?;

        let step_duration = Histogram::with_opts(
            HistogramOpts::new("nglab_step_duration_seconds", "Step execution time")
                .buckets(vec![0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1])
        )?;
        registry.register(Box::new(step_duration.clone()))?;

        let portfolio_value = IntGauge::new("nglab_portfolio_value", "Current portfolio value")?;
        registry.register(Box::new(portfolio_value.clone()))?;

        let orderbook_depth = IntGauge::new("nglab_orderbook_depth", "Order book depth")?;
        registry.register(Box::new(orderbook_depth.clone()))?;

        Ok(Self {
            orders_total,
            trades_total,
            step_duration,
            portfolio_value,
            orderbook_depth,
        })
    }
}
```

#### 11.2 Grafana Dashboard

**Create:** [monitoring/grafana/dashboards/nglab.json](monitoring/grafana/dashboards/nglab.json)
```json
{
  "dashboard": {
    "title": "NGLab Trading Metrics",
    "panels": [
      {
        "title": "Portfolio Value",
        "targets": [
          {
            "expr": "nglab_portfolio_value"
          }
        ],
        "type": "graph"
      },
      {
        "title": "Trades Per Second",
        "targets": [
          {
            "expr": "rate(nglab_trades_total[1m])"
          }
        ],
        "type": "graph"
      },
      {
        "title": "Step Duration (p99)",
        "targets": [
          {
            "expr": "histogram_quantile(0.99, nglab_step_duration_seconds_bucket)"
          }
        ],
        "type": "graph"
      }
    ]
  }
}
```

**Estimated Effort:** 3-4 days

---

### 12. Model Registry & Versioning

**Current State:** No systematic model versioning

#### 12.1 MLflow Integration

**Setup:** [python/src/tracking/mlflow_registry.py](python/src/tracking/mlflow_registry.py)
```python
import mlflow
from pathlib import Path
from typing import Dict, Any

class ModelRegistry:
    """MLflow-based model versioning and registry."""

    def __init__(self, tracking_uri: str = "sqlite:///mlflow.db"):
        mlflow.set_tracking_uri(tracking_uri)

    def log_model(
        self,
        model: Any,
        artifact_path: str,
        registered_model_name: str,
        metrics: Dict[str, float],
        hyperparameters: Dict[str, Any],
        tags: Dict[str, str] | None = None,
    ):
        """Log model with metrics and hyperparameters."""
        with mlflow.start_run():
            # Log hyperparameters
            mlflow.log_params(hyperparameters)

            # Log metrics
            mlflow.log_metrics(metrics)

            # Log tags
            if tags:
                mlflow.set_tags(tags)

            # Log model
            mlflow.pytorch.log_model(
                model,
                artifact_path=artifact_path,
                registered_model_name=registered_model_name,
            )

    def load_model(self, model_name: str, version: int | str = "latest"):
        """Load a registered model."""
        model_uri = f"models:/{model_name}/{version}"
        return mlflow.pytorch.load_model(model_uri)

    def transition_model_stage(
        self, model_name: str, version: int, stage: str
    ):
        """Transition model to a different stage (e.g., staging, production)."""
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage,
        )
```

**Estimated Effort:** 2-3 days

---

## Long-term Enhancements (P3)

> **Timeline:** 3-6 months
> **Impact:** Competitive advantages and advanced capabilities

### 13. Distributed Training Infrastructure

**Goal:** Scale training across multiple GPUs/nodes

#### 13.1 PyTorch DDP (Distributed Data Parallel)

**Update:** [python/src/pipeline/distributed_train.py](python/src/pipeline/distributed_train.py)
```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os

def setup_distributed():
    """Initialize distributed training."""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_distributed():
    """Clean up distributed training."""
    dist.destroy_process_group()

def train_distributed(model, dataloader, optimizer, epochs):
    """Distributed training loop."""
    local_rank = setup_distributed()

    model = model.to(local_rank)
    model = DDP(model, device_ids=[local_rank])

    for epoch in range(epochs):
        for batch in dataloader:
            # Training step
            pass

    cleanup_distributed()
```

**Launch:**
```bash
torchrun --nproc_per_node=4 python/src/pipeline/distributed_train.py
```

#### 13.2 Ray Tune for Hyperparameter Optimization

**Integration:** [python/src/pipeline/ray_tune.py](python/src/pipeline/ray_tune.py)
```python
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch

def train_function(config):
    """Training function for Ray Tune."""
    # Model training code
    pass

def run_hyperparameter_search():
    """Run distributed hyperparameter search."""
    search_space = {
        "lr": tune.loguniform(1e-5, 1e-1),
        "batch_size": tune.choice([16, 32, 64, 128]),
        "hidden_dim": tune.choice([64, 128, 256, 512]),
    }

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=100,
        grace_period=10,
    )

    search_alg = OptunaSearch()

    analysis = tune.run(
        train_function,
        config=search_space,
        scheduler=scheduler,
        search_alg=search_alg,
        num_samples=50,
        resources_per_trial={"cpu": 4, "gpu": 1},
    )

    return analysis.best_config
```

**Estimated Effort:** 1-2 weeks

---

### 14. Real-Time Model Serving

**Goal:** Deploy trained models as production APIs

#### 14.1 FastAPI Model Endpoint

**Create:** [python/src/api/inference.py](python/src/api/inference.py)
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np
from typing import List

app = FastAPI(title="NGLab Model Serving")

class PredictionRequest(BaseModel):
    """Input for model prediction."""
    observations: List[List[float]]
    temperature: float = 1.0

class PredictionResponse(BaseModel):
    """Model prediction output."""
    actions: List[int]
    probabilities: List[List[float]]
    latency_ms: float

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Generate predictions from trained model."""
    try:
        # Load model (cached)
        model = get_model()

        # Prepare input
        obs_tensor = torch.tensor(request.observations, dtype=torch.float32)

        # Inference
        with torch.no_grad():
            start = time.time()
            logits = model(obs_tensor)
            probs = torch.softmax(logits / request.temperature, dim=-1)
            actions = torch.argmax(logits, dim=-1)
            latency = (time.time() - start) * 1000

        return PredictionResponse(
            actions=actions.tolist(),
            probabilities=probs.tolist(),
            latency_ms=latency,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": model_loaded()}
```

**Launch:**
```bash
uvicorn src.api.inference:app --host 0.0.0.0 --port 8000 --workers 4
```

**Estimated Effort:** 4-5 days

---

### 15. Advanced Backtesting Framework

**Goal:** Historical simulation with realistic market conditions

#### 15.1 Event-Driven Backtesting Engine

**Create:** [python/src/backtesting/engine.py](python/src/backtesting/engine.py)
```python
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, List
import pandas as pd

@dataclass
class MarketEvent:
    """Market data update event."""
    timestamp: datetime
    symbol: str
    bid: float
    ask: float
    volume: float

class Strategy(Protocol):
    """Trading strategy interface."""

    def on_market_data(self, event: MarketEvent) -> List[Order]:
        """React to market data."""
        ...

    def on_fill(self, fill: Fill) -> None:
        """React to order fills."""
        ...

class BacktestEngine:
    """Event-driven backtesting engine."""

    def __init__(
        self,
        strategy: Strategy,
        initial_capital: float,
        commission: float = 0.001,
    ):
        self.strategy = strategy
        self.capital = initial_capital
        self.commission = commission
        self.portfolio = Portfolio()
        self.events: List[MarketEvent] = []

    def load_data(self, data: pd.DataFrame):
        """Load historical market data."""
        self.events = [
            MarketEvent(
                timestamp=row.Index,
                symbol=row.symbol,
                bid=row.bid,
                ask=row.ask,
                volume=row.volume,
            )
            for row in data.itertuples()
        ]

    def run(self) -> BacktestResults:
        """Execute backtest."""
        for event in self.events:
            # Process market event
            orders = self.strategy.on_market_data(event)

            # Execute orders
            for order in orders:
                fill = self.execute_order(order, event)
                if fill:
                    self.strategy.on_fill(fill)

        return self.calculate_results()

    def calculate_results(self) -> BacktestResults:
        """Calculate backtest metrics."""
        return BacktestResults(
            total_return=self.portfolio.total_return,
            sharpe_ratio=self.portfolio.sharpe_ratio,
            max_drawdown=self.portfolio.max_drawdown,
            win_rate=self.portfolio.win_rate,
            trades=self.portfolio.trades,
        )
```

**Estimated Effort:** 1 week

---

### 16. Live Trading Integration

**Goal:** Connect to real exchanges for paper/live trading

#### 16.1 Exchange Connector Interface

**Create:** [python/src/exchange/connector.py](python/src/exchange/connector.py)
```python
from abc import ABC, abstractmethod
from typing import List, Optional
import ccxt

class ExchangeConnector(ABC):
    """Abstract base class for exchange connections."""

    @abstractmethod
    def place_order(self, symbol: str, side: str, amount: float, price: Optional[float] = None):
        """Place order on exchange."""
        pass

    @abstractmethod
    def cancel_order(self, order_id: str):
        """Cancel existing order."""
        pass

    @abstractmethod
    def get_balance(self) -> dict:
        """Get account balance."""
        pass

    @abstractmethod
    def get_orderbook(self, symbol: str, depth: int = 20) -> dict:
        """Get order book snapshot."""
        pass

class BinanceConnector(ExchangeConnector):
    """Binance exchange connector using CCXT."""

    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        self.exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future' if testnet else 'spot',
            }
        })

        if testnet:
            self.exchange.set_sandbox_mode(True)

    def place_order(self, symbol: str, side: str, amount: float, price: Optional[float] = None):
        """Place order on Binance."""
        order_type = 'limit' if price else 'market'
        return self.exchange.create_order(
            symbol=symbol,
            type=order_type,
            side=side,
            amount=amount,
            price=price,
        )

    # Implement other methods...
```

**Estimated Effort:** 1-2 weeks

---

## Technical Debt

### 17. Code Quality Improvements

| Issue | Location | Priority | Effort |
|-------|----------|----------|--------|
| Remove `unwrap()` calls | [rust/src/**/*.rs](rust/src/) | High | 2 days |
| Add missing docstrings | [python/src/models/*.py](python/src/models/) | Medium | 3 days |
| Standardize error messages | All | Medium | 2 days |
| Remove commented code | All | Low | 1 day |
| Fix clippy warnings | [rust/](rust/) | Medium | 1 day |
| Fix mypy type errors | [python/](python/) | Medium | 2 days |

### 18. Dependency Updates

**Action:** Regular dependency audits and updates

**Create:** [.github/workflows/dependency-update.yml](.github/workflows/dependency-update.yml)
```yaml
name: Dependency Updates

on:
  schedule:
    - cron: '0 0 * * 0'  # Weekly on Sunday
  workflow_dispatch:

jobs:
  update-rust:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - name: Update Rust dependencies
        run: |
          cd rust
          cargo update
          cargo audit
      - name: Create PR
        uses: peter-evans/create-pull-request@v5
        with:
          title: "[Automated] Update Rust dependencies"
          branch: deps/rust-updates

  update-python:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
      - name: Update Python dependencies
        run: |
          cd python
          pip install pip-tools
          pip-compile --upgrade requirements.txt
      - name: Create PR
        uses: peter-evans/create-pull-request@v5
        with:
          title: "[Automated] Update Python dependencies"
          branch: deps/python-updates
```

---

## Performance Optimization Opportunities

### 19. Rust Performance

#### 19.1 SIMD Optimizations

**Target:** Order book matching algorithm

**Before:** [rust/src/simulation/orderbook.rs:150-200](rust/src/simulation/orderbook.rs:150-200)
```rust
// Standard iteration
for (price, orders) in &self.bids {
    // Process orders
}
```

**After (using SIMD):**
```rust
use std::simd::{f64x4, SimdFloat};

fn calculate_vwap_simd(prices: &[f64], volumes: &[f64]) -> f64 {
    let mut total_value = f64x4::splat(0.0);
    let mut total_volume = f64x4::splat(0.0);

    for chunk in prices.chunks_exact(4).zip(volumes.chunks_exact(4)) {
        let p = f64x4::from_slice(chunk.0);
        let v = f64x4::from_slice(chunk.1);
        total_value += p * v;
        total_volume += v;
    }

    total_value.reduce_sum() / total_volume.reduce_sum()
}
```

**Expected Improvement:** 2-4x for vectorizable operations

#### 19.2 Arena Allocation

**Use bump allocator for short-lived objects:**

```rust
use bumpalo::Bump;

pub struct TradingEnv {
    arena: Bump,
    // ... other fields
}

impl TradingEnv {
    pub fn step(&mut self, action: Action) -> StepResult {
        // Allocate temporary objects in arena
        let temp_data = self.arena.alloc(vec![0.0; 100]);

        // Process...

        // Arena is cleared at end of step
        self.arena.reset();
    }
}
```

**Expected Improvement:** Reduced GC pressure, 10-20% faster

### 20. Python Performance

#### 20.1 Numba JIT Compilation

**Target:** Feature engineering functions

```python
from numba import jit
import numpy as np

@jit(nopython=True)
def calculate_technical_indicators(prices: np.ndarray) -> np.ndarray:
    """Calculate indicators with JIT compilation."""
    n = len(prices)
    sma = np.zeros(n)
    ema = np.zeros(n)

    # SMA
    for i in range(20, n):
        sma[i] = np.mean(prices[i-20:i])

    # EMA
    alpha = 2.0 / 21.0
    ema[0] = prices[0]
    for i in range(1, n):
        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]

    return np.column_stack([sma, ema])
```

**Expected Improvement:** 10-100x for numerical code

#### 20.2 PyTorch Compilation (torch.compile)

**Update:** [python/src/pipeline/vae_module.py](python/src/pipeline/vae_module.py)
```python
import torch

class VAEModule(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model = VAE(config)

        # Compile model for faster execution
        if torch.__version__ >= "2.0.0":
            self.model = torch.compile(self.model, mode="reduce-overhead")
```

**Expected Improvement:** 20-50% training speedup

---

## Security Hardening

### 21. API Key Management

**Current State:** Keys in environment variables (good) but no rotation

#### 21.1 Secrets Management

**Use HashiCorp Vault or AWS Secrets Manager:**

**Create:** [python/src/utils/secrets.py](python/src/utils/secrets.py)
```python
import hvac
import os
from typing import Dict

class SecretsManager:
    """Manage secrets using HashiCorp Vault."""

    def __init__(self):
        self.client = hvac.Client(url=os.getenv("VAULT_ADDR"))
        self.client.token = os.getenv("VAULT_TOKEN")

    def get_secret(self, path: str) -> Dict[str, str]:
        """Retrieve secret from Vault."""
        response = self.client.secrets.kv.v2.read_secret_version(path=path)
        return response['data']['data']

    def rotate_api_key(self, exchange: str):
        """Rotate API key for exchange."""
        # Generate new key
        new_key = self.generate_api_key(exchange)

        # Update in Vault
        self.client.secrets.kv.v2.create_or_update_secret(
            path=f"nglab/{exchange}",
            secret={"api_key": new_key},
        )
```

### 22. Input Validation

**Add Pydantic validation for all API inputs:**

```python
from pydantic import BaseModel, Field, validator

class OrderRequest(BaseModel):
    """Validated order request."""

    symbol: str = Field(..., regex=r'^[A-Z]{2,10}$')
    side: Literal['buy', 'sell']
    quantity: float = Field(..., gt=0, le=1000000)
    price: Optional[float] = Field(None, gt=0)

    @validator('quantity')
    def validate_quantity(cls, v):
        if v <= 0:
            raise ValueError('Quantity must be positive')
        return v
```

### 23. Rate Limiting

**Add rate limiting to API endpoints:**

```python
from fastapi import FastAPI
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/predict")
@limiter.limit("10/minute")
async def predict(request: Request, data: PredictionRequest):
    # Prediction logic
    pass
```

---

## Documentation Improvements

### 24. Interactive Tutorials

**Create Jupyter notebooks:**

**Structure:**
```
docs/tutorials/
├── 01_getting_started.ipynb
├── 02_training_your_first_model.ipynb
├── 03_custom_trading_strategies.ipynb
├── 04_hyperparameter_tuning.ipynb
├── 05_production_deployment.ipynb
└── 06_advanced_features.ipynb
```

### 25. API Reference

**Auto-generate from code:**

```bash
# Rust
cargo doc --no-deps --open

# Python
sphinx-apidoc -o docs/source python/src
sphinx-build -b html docs/source docs/build

# TypeScript
typedoc --out docs/typescript src/
```

### 26. Architecture Decision Records (ADRs)

**Create:** [docs/adr/](docs/adr/)
```
docs/adr/
├── 0001-use-rust-for-simulation.md
├── 0002-pytorch-as-ml-framework.md
├── 0003-tauri-for-desktop-app.md
├── 0004-gymnasium-compatible-interface.md
└── template.md
```

**Template:** [docs/adr/template.md](docs/adr/template.md)
```markdown
# ADR-XXXX: [Title]

## Status
[Proposed | Accepted | Deprecated | Superseded]

## Context
[What is the issue that we're seeing that is motivating this decision or change?]

## Decision
[What is the change that we're proposing and/or doing?]

## Consequences
[What becomes easier or more difficult to do because of this change?]

## Alternatives Considered
[What other options did we evaluate?]
```

---

### 27. Architecture Decision Records (ADRs) (Completed)
**Status:** ✅ Structure and initial ADRs created in `docs/adr/`.

### 28. Distributed Tracing (Completed)
**Status:** ✅ OpenTelemetry/Jaeger integrated into Rust engine.

### 29. Interactive Tutorials (Completed)
**Status:** ✅ "Getting Started" notebook created in `docs/tutorials/`.

## Implementation Roadmap

### Phase 1: Foundation (Weeks 1-4)

**Critical Path:**
1. ✅ Frontend testing infrastructure (Week 1)
2. ✅ Error handling & logging (Week 2)
3. ✅ Configuration management (Week 3)
4. ✅ Health checks & monitoring (Week 4)

**Success Criteria:**
- 80% test coverage on frontend hooks
- Zero unwrap() calls in production paths
- Structured logging operational
- Health check endpoints responding

---

### Phase 2: Production Readiness (Weeks 5-12)

**Parallel Tracks:**

**Track A: Documentation (Weeks 5-7)**
- API documentation complete
- Interactive tutorials published
- Architecture decision records started

**Track B: Infrastructure (Weeks 5-8)**
- Docker containerization
- CI/CD deployment pipeline
- Database integration

**Track C: Observability (Weeks 9-12)**
- Prometheus metrics
- Grafana dashboards
- Distributed tracing

**Success Criteria:**
- Docker build under 5 minutes
- Automated deployments to staging
- Real-time metrics dashboard
- Complete API documentation

---

### Phase 3: Advanced Features (Weeks 13-24)

**Major Initiatives:**

**Distributed Training (Weeks 13-16)**
- PyTorch DDP setup (Planned)
- Ray Tune integration (In-progress)
- Multi-GPU benchmarks (Planned)

**Model Serving, Logging & CLI (Weeks 17-20)** (Completed)
- ✅ FastAPI inference endpoints
- ✅ Model registry (MLflow)
- ✅ Logit Lens & Enhanced Visualizations
- ✅ Modular CLI Framework

**Production Trading (Weeks 21-24)**
- Backtesting engine
- Exchange connectors
- Risk management system

**Success Criteria:**
- 4x training speedup on 4 GPUs
- <10ms inference latency (p99)
- Backtest 1M ticks in <1 minute
- Paper trading operational

---

### Phase 4: Optimization & Scale (Months 7-12)

**Performance:**
- SIMD optimizations
- Profile-guided optimization
- Memory pool allocations

**Scale:**
- Kubernetes deployment
- Auto-scaling policies
- Multi-region setup

**Advanced ML:**
- Multi-agent learning
- Meta-learning for strategy adaptation
- Ensemble models

---

## Metrics & KPIs

### Code Quality Targets

| Metric | Current | Target |
|--------|---------|--------|
| Test Coverage (Python) | ~60% | >80% |
| Test Coverage (TypeScript) | 0% | >70% |
| Clippy Warnings | ~10 | 0 |
| Mypy Errors | Unknown | 0 (strict mode) |
| Documentation Coverage | 60% | >90% |

### Performance Targets

| Metric | Current | Target |
|--------|---------|--------|
| Order Book Matching | ~50K ops/s | >100K ops/s |
| Env Step Time | <1ms | <0.5ms |
| Model Inference (p99) | Unknown | <10ms |
| Training Throughput | Baseline | 4x on 4 GPUs |

### Reliability Targets

| Metric | Target |
|--------|--------|
| API Uptime | 99.9% |
| Error Rate | <0.1% |
| P99 Latency | <100ms |
| Mean Time to Recovery | <5 min |

---

## Resource Requirements

### Development Team

**Phase 1-2 (16 weeks):**
- 1 Senior Backend Engineer (Rust/Python)
- 1 Frontend Engineer (TypeScript/React)
- 1 DevOps Engineer (part-time, 50%)
- 1 ML Engineer (part-time, 25%)

**Phase 3-4 (40 weeks):**
- Add 1 ML Engineer (full-time)
- Add 1 QA Engineer (part-time, 50%)

### Infrastructure Costs (Estimated)

**Development/Staging:**
- CI/CD runners: $100/month
- Cloud compute: $200/month
- Database: $50/month
- Monitoring: $50/month

**Production:**
- Compute (4x GPU instances): $2,000/month
- Database (managed PostgreSQL): $200/month
- Monitoring & Logging: $150/month
- Data storage: $100/month

**Total Year 1:** ~$35,000

---

## Risks & Mitigations

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| PyO3 version incompatibility | High | Low | Pin versions, comprehensive tests |
| Performance regression | Medium | Medium | Continuous benchmarking in CI |
| Model overfitting | High | High | Cross-validation, out-of-sample testing |
| Exchange API changes | Medium | Medium | Abstraction layer, adapter pattern |
| Security breach | High | Low | Security audits, penetration testing |
| Data quality issues | High | Medium | Data validation, anomaly detection |

---

## Conclusion

NGLab has a solid architectural foundation and demonstrates strong engineering practices. The improvements outlined in this document will transform it from an early-stage prototype into a production-ready trading platform.

**Immediate Priorities:**
1. Frontend testing (P0)
2. Production error handling (P0)
3. Configuration management (P0)
4. API documentation (P1)
5. Containerization (P1)

**Success Timeline:**
- **Month 3:** Production-ready core platform
- **Month 6:** Advanced ML features operational
- **Month 12:** Scalable, distributed system with live trading capabilities

**Next Steps:**
1. Review and prioritize improvements with stakeholders
2. Create detailed technical specifications for P0 items
3. Set up project tracking (GitHub Projects/Jira)
4. Begin Phase 1 implementation

---

**Document Version:** 1.0
**Last Updated:** 2026-01-16
**Maintainer:** Development Team
**Review Cycle:** Quarterly

For questions or contributions, please refer to [CONTRIBUTING.md](CONTRIBUTING.md).
