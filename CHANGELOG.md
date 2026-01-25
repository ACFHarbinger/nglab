# Changelog

All notable changes to the NGLab project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- Comprehensive documentation enhancements (DEVELOPMENT.md, TESTING.md)
- Enhanced AGENTS.md with agent comparison matrix and reward library
- Enhanced ARCHITECTURE.md with Kubernetes topology
- Enhanced TROUBLESHOOTING.md with GPU/CUDA debugging guides
- Interactive Jupyter notebook tutorial series (10 notebooks)

---

## [0.10.0] - 2026-01-25

> **Codename**: "Stability & Agents"

### Added

#### Market Stability
- **Periodic Auctions**: Opening, Closing, and Volatility auction phases via `AuctionState`.
- **Circuit Breakers**: Infrastructure for volatility auctions to prevent flash crashes.
- **Order Expiration**: Good-Till-Date (GTD) support with `prune_expired_orders`.
- **Advanced Order Types**: Full Bracket Order support (Entry + SL + TP) with OCO linking.

#### Multi-Agent Simulation
- **Agent Framework**: Extensible `Agent` trait in `simulation/multi_agent.rs` (`agents.rs`).
- **Standard Agents**:
  - `NoiseAgent`: Provides baseline liquidity.
  - `MomentumAgent`: Trend-following strategy.
- **Agent Manager**: Orchestration for multi-agent market scenarios via `AgentManager`.
- **Performance Tracking**: `TradeRecord` attribution for agent analytics.

#### Options Simulation
- **Options Market**: New `OptionsMarket` managing multiple CLOBs for derivatives.
- **Pricing Engine**: Integrated `black_scholes` for real-time Greeks (Delta, Gamma, Vega).
- **Lifecycle**: Automatic expiry and exercise logic for ITM contracts.

#### Scenario Analysis
- **Scenario Engine**: Deterministic `Scenario` types (PriceShock, VolatilitySpike) for stress testing.
- **Monte Carlo**: Geometric Brownian Motion (GBM) path generator for VaR and CVaR calculation.


### Fixed
- **Order Matching**: Consolidated `match_order` logic to correctly handle maker-side OCO/Bracket triggers.
- **Microstructure**: Proper maker fill tracking and iceberg order replenishment.

---

## [0.9.0] - 2026-01-20

> **Codename**: "Testing Hardening"

### Added

#### Forecasting Model Integration
- **Robust ARIMA**: Yule-Walker equation solving in Rust (`rust/src/moon/arima.rs`)
- **Prophet Configuration**: UI controls for growth, seasonality, and component flags
- **Numerical Stability**: `safe_div` and `SafeFloat` utilities in `rust/src/utils/math.rs`
- **Validation**: Zod schemas for all forecasting model parameters

#### Testing Infrastructure
- Comprehensive unit tests for **Differential Evolution (DE)** and HPO wrappers
- End-to-end tests for `infer.py` script
- `hpo_fixtures.py` and `model_fixtures.py` for mock model artifacts
- `TradingEnv` and `PolymarketEnv` test suites
- `nglab_bindings.py` tests for Rust PyO3 integration

### Fixed
- **Inference Script**: Module resolution errors in `python/src/infer.py`
- **Rust Lints**: Unused import and variable warnings in simulation modules

---

## [0.8.0] - 2026-01-19

> **Codename**: "Multi-Asset Trading"

### Added

#### Multi-Asset Environment
- `MultiAssetEnv` in Rust for simultaneous multi-asset simulation
- Native `reset_native` and `step_native` API for Rust-side simulations
- Zero-copy observation transfers to Python
- Feature generation: prices, log returns, volatility, order book imbalance

#### Advanced Order Types
- **Iceberg Orders**: Hidden quantities with automatic visible slice refill
- **Trailing Stop Orders**: Dynamic trigger prices with `trailing_delta` config
- **Order Modification**: `modify_order` API with time-priority preservation

#### Algorithmic Execution
- **TWAP**: `AlgoOrder` tracking and slicing logic for automated execution
- **Synthetic Liquidity**: Realistic slippage with stochastic 0-0.1% variance

#### Risk Management Integration
- `RiskManager` integrated into `MultiAssetEnv` and `TradingEnv`
- Real-time tracking: Risk Score, Current Drawdown, Value at Risk (VaR)
- Automatic position sizing via `position_multiplier`
- Risk metrics exposed to Python via PyO3

#### Machine Learning Features
- **Feature Selection**: `TimeSeriesFeatureSelector` with MI scoring and RFECV
- **Online Learning**: `OnlineTrainer` with incremental `partial_fit` updates
- **Portfolio Optimization**: Markowitz Mean-Variance and HRP algorithms
- **Risk Parity**: Inverse volatility allocation

#### Debug & Observability
- **Risk Dashboard**: `RiskDashboardWidget` for real-time monitoring
- **Global Debug Mode**: Toggle in Tauri backend with UI switch
- **Production Source Maps**: Enabled in Vite configuration
- **Memory Profiling**: `@timeit` and `@memory_profile` decorators

### Fixed
- **Dependency Resolution**: Python version limited to `>=3.11, <3.13`
- **Rust Bindings**: PyO3 attribute scope issues in `RiskConfig`

---

## [0.7.0] - 2026-01-18

> **Codename**: "Infrastructure & Scale"

### Added

#### Horizontal Scaling
- **Ray Serve**: Distributed model inference
- **PgBouncer**: Connection pooling for PostgreSQL
- **Read Replica**: PostgreSQL replica configuration

#### Distributed Tracing
- **OpenTelemetry**: Parent-Based Ratio Sampling (10%)
- **Jaeger**: Full trace visualization integration
- **Custom Spans**: Batch processing bottleneck detection

#### Kubernetes Deployment
- **Canary Deployments**: Gradual rollout templates
- **HPA**: CPU/memory-based horizontal pod autoscaling
- **ts-rs**: Automatic Rust→TypeScript type generation

#### Database Optimization
- 10+ strategic indexes for common query patterns
- Redis query caching layer with TTL
- Data archival system with automated scripts
- postgres_exporter for monitoring
- Automated backup with S3/GCS support

#### Alerting
- 30+ production-grade AlertManager rules
- Anomaly detection using statistical methods
- Alert correlation with 5 inhibition rules

### Fixed
- **Inference Service**: Syntax errors in Ray Serve deployment
- **Agent Evaluation**: Relative import error for `ContinuousActionWrapper`
- **Rust Documentation**: Doctest import paths corrected

---

## [0.6.0] - 2026-01-18

> **Codename**: "Frontend Testing"

### Added

#### Vitest Integration
- Setup with `@testing-library/react` and `jsdom`
- Mocks for Tauri APIs, `ResizeObserver`, `lightweight-charts`

#### Component Tests (28 files, 115 cases)
- **Dashboard**: GlobalActivityWidget, UserProfileWidget, OrderBook
- **Terminal**: TerminalLayout, TerminalChart, OrderBookWidget
- **Charts**: PriceChart data updates and lifecycle
- **Tabs**: All major tabs tested (Scraper, Analysis, Prediction, etc.)
- **Hooks**: useArena, usePolymarket, useFavorites

#### Cypress E2E
- 13 test spec files covering all major components
- Tauri API mocking with default responses
- Custom commands and market data fixtures

### Changed
- **Favorites**: Integrated `useFavorites` hook into Terminal
- **Multi-Outcome**: Updated TradingFormWidget for multi-outcome markets

---

## [0.5.0] - 2026-01-17

> **Codename**: "Documentation & Quality"

### Added

#### Comprehensive Documentation
- JSDoc for entire TypeScript frontend
- Rust inner-module style (`//!`) documentation
- Python PEP 257 docstrings for all modules
- CSS file documentation

#### Classical ML Models (Expanded)
- **Regression**: LARS, Stepwise, M5, MARS, LOESS
- **Trees**: CART, ID3, C4.5, C5.0, CHAID, DecisionStump
- **Ensemble**: AdaBoost, Bagging, Stacking, Voting, GBRT
- **Bayesian**: GaussianNB, MultinomialNB, AODE, BBN
- **SVM**: LinearSVM, NuSVM, OneClassSVM, LS-SVM, TWSVM
- **Clustering**: K-Medians, EM (GMM)
- **Association**: Eclat, Apriori, FPGrowth
- **Reduction**: PCR, PLSR, MDS, Sammon, UMAP

#### CI/CD & Code Quality
- `Black`, `Ruff`, `MyPy`, `Pip-Audit`, `Pytest-Cov` integration
- `Prettier` for TypeScript/JavaScript
- `ktlint` for Android Kotlin
- Strict language isolation in pre-commit

### Fixed
- Build environment linker issues
- Deep learning model import errors
- Test stability with shape mismatches

---

## [0.4.0] - 2026-01-16

> **Codename**: "Neural Networks"

### Added

#### Advanced Architectures
- **Memory**: DNC, NTM, Hopfield Network
- **Recurrent**: LSM, ESN, ELM
- **Convolutional**: DCN, DN, DCIGN, DRN
- **Generative**: VAE, TimeGAN, Diffusion UNet, RealNVP
- **Physics**: NODE, PINN
- **Spiking**: SNN with custom LIFCell
- **Attention**: Multi-head self-attention with positional encoding

#### Frontend Tabs
- **News Tab**: Aggregating customized news feeds
- **Training Tab**: Neural network training interface
- **Prediction Tab**: Deep learning model inference

#### Backend Commands
- `list_trained_models` and `predict_trained_model` Tauri commands
- `infer.py` for standalone model inference

---

## [0.3.0] - 2026-01-15

> **Codename**: "Production Infrastructure"

### Added

#### Docker & Kubernetes
- Multi-stage `Dockerfile.prod` (~500MB optimized image)
- `Dockerfile.gpu` with CUDA 12.1 support
- `docker-compose.prod.yml` with full stack
- Kustomize base and overlays (dev/staging/prod)
- Helm charts with configurable values

#### Monitoring Stack
- Prometheus v2.48 with 15-day retention
- Grafana 10.2 with provisioned dashboards
- AlertManager v0.26 with severity routing
- Jaeger 1.52 for distributed tracing

#### CI/CD Pipeline
- Multi-architecture Docker builds (amd64, arm64)
- Staging/production deployments
- Smoke tests and rollout verification
- Slack notifications

#### Model Storage Backends
- **LocalStorage**: File-based versioning
- **S3Storage**: boto3 with async support
- **GCSStorage**: google-cloud-storage
- zstd/gzip compression
- Local caching for cloud backends

#### GPU Optimization
- `CUDAProfiler` with Chrome trace export
- TensorBoard trace handler
- Memory profiling with stack traces
- GPU benchmark suite with latency percentiles

#### Mixed Precision Training
- `MixedPrecisionTrainer` wrapper
- FP16-mixed, BF16-mixed, FP32 modes
- Hardware-aware precision selection
- Memory savings estimation

---

## [0.2.0] - 2026-01-14

> **Codename**: "Performance & Scaling"

### Added

#### GPU Acceleration
- `GPUFeatureEngineer` for 100x speedup in indicator calculation
- Technical indicators: SMA, EMA, RSI, MACD, Bollinger Bands

#### Environment Batching
- `VectorizedTradingEnv` for parallel simulation
- `SubprocVecEnv` for multi-process parallelism
- TorchRL integration with `ParallelEnv`

#### Online Learning
- `PageHinkley` and `MovingAverageDrift` detectors
- `OnlineNormalizer` using Welford's algorithm

#### FastAPI Scaling
- `BatchInferenceHandler` for async predictions
- Redis caching layer
- Production Gunicorn config

---

## [0.1.0] - 2026-01-13

> **Codename**: "Foundation"

### Added

#### Core Simulation (Rust)
- `TradingEnv` with Gymnasium-compatible interface
- `OrderBook` with price-time priority CLOB
- `PolymarketArena` for prediction market simulation
- PyO3 bindings with zero-copy NumPy transfers

#### ML Pipeline (Python)
- Time-series forecasting models (ARIMA, GARCH, Prophet)
- Deep learning backbones (Mamba, Transformer, LSTM)
- VAE for regime detection
- TorchRL integration for RL agents

#### Frontend (TypeScript/Tauri)
- Dashboard with widgets
- Terminal trading interface
- Real-time price charts
- Order book visualization

#### DevOps
- GitHub Actions CI/CD
- Justfile task automation
- Pre-commit hooks

---

## Version Roadmap

| Version | Target | Focus |
|---------|--------|-------|
| 0.10.0 | Q1 2026 | Multi-agent RL simulation |
| 0.11.0 | Q2 2026 | Distributed training with Ray |
| 1.0.0 | Q3 2026 | Production-ready release |

---

## Links

- **Repository**: [github.com/acfharbinger/nglab](https://github.com/acfharbinger/nglab)
- **Documentation**: [TUTORIAL.md](TUTORIAL.md)
- **Issues**: [Report a Bug](https://github.com/acfharbinger/nglab/issues)

---

[Unreleased]: https://github.com/acfharbinger/nglab/compare/v0.9.0...HEAD
[0.9.0]: https://github.com/acfharbinger/nglab/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/acfharbinger/nglab/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/acfharbinger/nglab/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/acfharbinger/nglab/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/acfharbinger/nglab/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/acfharbinger/nglab/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/acfharbinger/nglab/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/acfharbinger/nglab/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/acfharbinger/nglab/releases/tag/v0.1.0
