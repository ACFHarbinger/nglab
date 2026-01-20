# Changelog

All notable changes to the NGLab project will be documented in this file.

## [Unreleased] - 2026-01-20

### Added

- **HPO & Inference Test Suite**:
  - Implemented comprehensive unit tests for **Differential Evolution (DE)** and HPO optimization wrappers in `python/tests/unit/test_hpo.py`.
  - Created end-to-end tests for the `infer.py` script in `python/tests/unit/test_infer.py`, verifying JSON prediction responses and error handling.
  - Developed `hpo_fixtures.py` and `model_fixtures.py` for automated mock model artifact generation with full `ModelMetadata`.
  - Standardized `conftest.py` to modularly load new test fixtures.
- **Environment & Bindings Test Suite**:
  - Implemented comprehensive unit tests for `TradingEnv` and `PolymarketEnv` in `python/tests/test_trading_env.py` and `python/tests/test_polymarket_env.py`.
  - Created `nglab_bindings.py` tests to verify Rust PyO3 integration and optional return type handling.
  - Added `test_env_integration.py` to verify consistency between Python wrappers and Rust backend, including fallback mechanisms.
  - Created modular fixtures in `environment_fixtures.py` and `nglab_fixtures.py` for environment and market data simulation.

## [Unreleased] - 2026-01-19

### Added

- **Advanced Trading Features (Phase 4)**:
  - **Multi-Asset Environment**: Implemented `MultiAssetEnv` in Rust (`rust/src/simulation/multi_asset.rs`) supporting simultaneous simulation of multiple assets.
    - Native `reset_native` and `step_native` API for Rust-side simulations.
    - Python bindings with zero-copy observation transfers.
    - Comprehensive feature generation: prices, log returns, volatility (rolling std dev), order book imbalance, and normalized positions.
  - **Iceberg Orders**: Implemented logic in `OrderBook` to handle hidden quantities.
    - Automatic visible slice refill from total iceberg quantity.
    - Time-priority reset (moved to back of queue) upon each refill for realistic simulation.
  - **Trailing Stop Orders**: Added support for dynamic trigger prices.
    - Flexible `trailing_delta` configuration for both Bid and Ask sides.
    - Real-time trigger price adjustment tracking market movements.
  - **Improved Execution Matching**: 
    - Replaced simple tape-price matching with actual Order Book matching in `MultiAssetEnv`.
    - Added synthetic liquidity seeding around tape prices for realistic slippage.
    - Integrated stochastic slippage (0-0.1%) using environment RNG.
  - **Unit Testing**: Added comprehensive tests for Iceberg refills, Trailing Stop dynamics, and multi-asset environment loops.
- **Advanced ML & Portfolio Features (Phase 5)**:
  - **Automated Feature Selection**: Implemented `TimeSeriesFeatureSelector` in Python with Support for Mutual Information (MI) scoring and Recursive Feature Elimination (RFECV).
  - **Order Modification**: Implemented `modify_order` API in `OrderBook` with time-priority preservation for quantity reductions and automatic re-submission for price changes.
  - **Algorithmic Execution (TWAP)**: Added `AlgoOrder` tracking and slicing logic in `MultiAssetEnv` for automated trade execution over time.
  - **Online Learning Infrastructure**: Created `OnlineTrainer` with support for incremental `partial_fit` updates and experience replay for real-time model adaptation.
  - **Portfolio Optimization Layer**: Added `PortfolioOptimizer` with support for Markowitz Mean-Variance and Hierarchical Risk Parity (HRP) algorithms.
  - **Pipeline Integration**: Seamlessly integrated advanced selection methods into the `FeaturePipeline` for automated dimensionality reduction.
- **Risk Management Integration (Phase 4.3)**:
  - Integrated `RiskManager` into the core Rust simulation types (`MultiAssetEnv` and `TradingEnv`).
  - Added real-time tracking of **Risk Score**, **Current Drawdown**, and **Value at Risk (VaR)**.
  - Implemented automatic position sizing limits via `position_multiplier` that scales down exposure on limit breaches (drawdown, daily loss, VaR).
  - Exposed risk metrics and configuration to Python via PyO3 bindings.
  - Added native Rust unit tests for risk integration (`test_risk_integration`).
- **Portfolio Optimization Enhancements**:
  - Implemented **Risk Parity (Inverse Volatility)** allocation in `PortfolioOptimizer`.
  - Added automated calculation of risk-equalizing weights for multi-asset portfolios.
- **Debug & Observability (Phase 5.3)**:
  - **Risk Dashboard**: Created a dedicated `RiskDashboardWidget` in the React frontend for real-time risk monitoring.
  - **Global Debug Mode**: Implemented a global `debug_mode` toggle in the Tauri backend with a dedicated toggle switch in the Account/Settings UI.
  - **Extended Debug Infrastructure**:
    - Enabled **production source maps** in Vite configuration.
    - Configured Rust **debug symbols** in the release profile for better production troubleshooting.
    - Added **`@timeit`** and **`@memory_profile`** decorators in Python for high-performance timing and memory tracking.
    - Implemented a real-time **`get_memory_usage`** Tauri command to monitor process RSS and VMS.
  - **Telemetry**: Extended the `ArenaUpdate` event to include real-time risk status for live visualization.
- **Infrastructure & Scalability (Phase 6)**:
  - **Horizontal Scaling**: Migrated model inference to **Ray Serve**, enabling distributed, scalable prediction serving.
  - **Database Availability**: Integrated **PgBouncer** for connection pooling and added a **PostgreSQL Read Replica** configuration in `docker-compose.prod.yml`.
  - **Distributed Tracing**: Enhanced OpenTelemetry instrumentation with **Parent-Based Ratio Sampling** (10%) and custom spans for batch processing bottlenecks.
  - **Observability Stack**: Fully integrated **Jaeger** for trace visualization and updated **AlertManager** with 20+ production-grade rules (latency, error rates, GPU health).
  - **Deployment**: Created **Kubernetes Canary Deployment** templates and refactored CI/CD for robust multi-layer linting and testing.
  - **TypeScript Bindings**: Integrated **ts-rs** for automatic Rust→TypeScript type generation, enabling type-safe Tauri command interfaces.
  - **Database Optimization**:
    - Created strategic indexes for common query patterns (10+ indexes)
    - Implemented Redis query caching layer with decorators and TTL
    - Built data archival system with dedicated tables and automated scripts
    - Integrated postgres_exporter for comprehensive database monitoring
    - Automated backup system with S3/GCS support and retention policies
  - **Advanced Alerting**:
    - Anomaly detection alerts using statistical methods (rate-of-change, std dev)
    - Alert correlation and intelligent grouping with 5 inhibition rules
    - Expanded AlertManager with 30+ production-grade alert rules
  - **CI/CD Improvements**: 
    - Added ESLint configuration with TypeScript support
    - Implemented `npm run lint` and `npm run test` scripts in root package.json
    - Fixed all TypeScript lint errors across frontend test suite
    - Resolved Rust doctest compilation issues
    - Verified all 44 tests pass successfully

### Fixed

- **Dependency Resolution**: Fixed a `uv` dependency resolution error by limiting supported Python versions to `>=3.11, <3.13` in `pyproject.toml`, ensuring compatibility with OpenTelemetry and other ML libraries.
- **Rust Bindings**: Resolved PyO3 attribute scope issues in `RiskConfig` and `RiskStatus` by standardizing on `pyclass(get_all, set_all)`.
- **Inference Service**: Fixed syntax and logic errors in `inference.py` related to OpenTelemetry instrumentation and Ray Serve deployment guards.
- **Agent Evaluation**: Fixed a relative import error in `evaluate_agents.py` by using the absolute package path for `ContinuousActionWrapper`.
- **Rust Documentation**: Fixed doctest import path from `nglab::simulation::TradingEnv` to `nglab::simulation::gym::TradingEnv` and added missing seed parameter.
- **Frontend Linting**: 
  - Removed unused variables across TypeScript test files
  - Fixed `ArenaUpdate` type mismatch in PriceChart tests by adding risk metrics fields
  - Cleaned up unused imports from testing library components
  - Added `_` prefix pattern to ESLint for intentionally unused function parameters
  - **Troubleshooting Guide**: Created `TROUBLESHOOTING.md` covering common simulation, environment, and frontend issues.
  - **VS Code Integration**: Added `.devcontainer/devcontainer.json` for standardized development environments.
  - **Mock Data Generation**: Implemented `script/seed_data.py` for generating high-fidelity GBM-based market data.
  - **Justfile Enhancements**: Added `seed-data` and consolidated `docs` recipes to the project `justfile`.

- **Production Deployment Infrastructure (Phase 3)**:
  - Created **docker-compose.prod.yml** with full production stack:
    - Scaled API service with health checks and resource limits
    - GPU API service with NVIDIA container runtime support
    - PostgreSQL 16 with health checks and persistence
    - Redis 7 for caching with AOF persistence
    - Prometheus v2.48 with 15-day retention and alert rules
    - Grafana 10.2 with provisioned datasources and dashboards
    - AlertManager v0.26 with severity-based routing
    - Jaeger 1.52 for distributed tracing (Badger storage)
    - Nginx reverse proxy with rate limiting and SSL termination
  - Created **Dockerfile.prod** - multi-stage optimized production image (~500MB)
  - Created **Dockerfile.gpu** - CUDA 12.1 enabled GPU inference image
  - **Kubernetes Deployment** (deploy/k8s/):
    - Kustomize base with namespace, configmaps, secrets
    - Deployments for API and GPU API with proper resource limits
    - HorizontalPodAutoscaler with CPU/memory scaling
    - Services, Ingress with TLS, PersistentVolumeClaims
    - ServiceAccount with RBAC for pod access
    - Production overlay with replica scaling
  - **Helm Charts** (deploy/helm/nglab/):
    - Full chart with configurable values for all services
    - PostgreSQL and Redis as Bitnami dependencies
    - Secrets management and ingress configuration
    - GPU node selector and tolerations support
  - **CI/CD Pipeline** (.github/workflows/deploy.yml):
    - Multi-architecture Docker builds (amd64, arm64)
    - Staging deployment with Kustomize
    - Production deployment with Helm
    - Smoke tests and rollout verification
    - Slack notifications on failure
  - **Monitoring Configuration** (monitoring/):
    - Prometheus scrape configs and alert rules (API latency, error rates, GPU metrics, database, Redis, trading alerts)
    - AlertManager routing with severity-based escalation
    - Grafana datasource provisioning (Prometheus, Jaeger)
    - Dashboard provisioning configuration
  - **Nginx Configuration** (deploy/nginx/):
    - Load balancer with upstream pools
    - Rate limiting (100 req/s with burst)
    - SSL/TLS configuration with modern ciphers
    - WebSocket support for streaming endpoints
    - Separate routing for GPU inference endpoints

- **Model Storage Backends** (python/src/storage/):
  - Created unified **ModelStorage** abstract interface with versioning support
  - Implemented **LocalStorage** backend with file-based versioning
  - Implemented **S3Storage** backend with boto3 (async-capable)
  - Implemented **GCSStorage** backend with google-cloud-storage
  - Added zstd/gzip compression for checkpoint transfer
  - Automatic version cleanup (configurable max_versions)
  - Local caching layer for cloud backends
  - Factory function `create_storage()` with environment detection

- **GPU Profiling & Optimization** (python/src/utils/profiling/):
  - Created **CUDAProfiler** class with torch.profiler integration:
    - Chrome trace export for visualization
    - TensorBoard trace handler support
    - Configurable schedule (wait, warmup, active steps)
    - Memory profiling with stack traces and FLOPS
  - Added `profile_model_forward()` utility for inference profiling
  - Added `profile_training_step()` utility for full training step analysis
  - Created `get_gpu_memory_stats()` for real-time memory monitoring
  - **GPU Benchmark Suite** (benchmark.py):
    - `GPUBenchmark` class for standardized performance testing
    - Inference benchmarks across batch sizes
    - Training benchmarks with optimizer step timing
    - P50/P95/P99 latency percentile metrics
    - Throughput (samples/sec, batches/sec) calculation
    - Baseline comparison for regression detection
    - JSON export for CI integration

- **Mixed Precision Training** (python/src/utils/mixed_precision.py):
  - Created **MixedPrecisionTrainer** wrapper class:
    - Automatic GradScaler management for FP16
    - Support for FP16-mixed, BF16-mixed, and FP32 modes
    - Gradient clipping with proper unscaling
    - State dict save/load for checkpointing
  - Added `get_optimal_precision()` for hardware-aware configuration:
    - Ampere+ GPUs: BF16-mixed (SM 8.0+)
    - Volta/Turing GPUs: FP16-mixed (SM 7.0+)
    - Older GPUs: FP32
  - Created `estimate_memory_savings()` for planning memory reduction
  - Added `configure_model_for_mixed_precision()` for layer-specific dtype handling

- **Prefetching DataLoader** (python/src/data/prefetch_dataloader.py):
  - Created **CUDAPrefetcher** using CUDA streams for async GPU transfer
  - Created **BackgroundPrefetcher** with threading for CPU-bound loading
  - Extended **PrefetchDataLoader** from PyTorch DataLoader:
    - Automatic pinned memory for GPU training
    - Configurable prefetch_factor and persistent_workers
    - Device-aware iterator with CUDA prefetching
  - Added `create_optimized_dataloader()` factory with best practices
  - Created `benchmark_dataloader()` for throughput measurement

- **Environment Configuration**:
  - Updated **.env.example** with comprehensive configuration:
    - Database (PostgreSQL) and cache (Redis) settings
    - GPU configuration (CUDA_VISIBLE_DEVICES, architectures)
    - Model storage (local, S3, GCS) with credentials
    - Monitoring (Prometheus, Grafana) settings
    - Tracing (OpenTelemetry, Jaeger) configuration
    - Mixed precision training settings
    - Distributed training (DDP) configuration

- **Developer Experience (Quick Wins)**:
  - Created comprehensive **Makefile** with colorized output and emojis for common development tasks (`make test`, `make build`, `make lint`, `make fmt`, `make clean`, `make dev`, `make run-tauri`).
  - Added **.env.example** documenting all environment variables (WANDB, MLflow, CUDA, data paths, cloud storage, training configs).
  - Added **pytest GPU markers** with automatic skip logic for tests requiring CUDA (`@pytest.mark.gpu`).
  - Enhanced `conftest.py` with `cuda_available` fixture and `pytest_collection_modifyitems` hook for GPU test management.
- **RNG Seeding for Reproducibility**:
  - Added `StdRng` field to `TradingEnv` for deterministic random number generation.
  - Added optional `seed: Option<u64>` parameter to `TradingEnv::new()` and PyO3 `new_py()` constructors.
  - Implemented `set_seed(&mut self, seed: u64)` method for runtime re-seeding.
  - Updated `reset()` method to properly initialize RNG from seed parameter.
  - Updated all Rust tests, benchmarks, and Tauri initialization to use new seed API.
- **DataLoader Infrastructure for Non-RL Tasks**:
  - Created `FinancialDataset` class extending `TimeSeriesDataset` with financial preprocessing support.
  - Implemented `create_dataloader()` factory function supporting CSV, Parquet, and HDF5 formats.
  - Added proper train/val/test splits with shared normalization statistics.
  - Created `conf/data/dataloader.yaml` Hydra config for dataloader settings.
  - Integrated DataLoader into `main.py` replacing stub implementation.
- **Health Check Integration (Tauri + React)**:
  - Created `commands/health.rs` with `health_check()` and `get_system_info()` Tauri commands.
  - Implemented `HealthStatus` struct with component-level health monitoring (Arena, OrderBook, Polymarket, Python bindings).
  - Added uptime tracking and system info (CPU count, OS, architecture).
  - Created `HealthDashboard.tsx` React widget with auto-refresh, status indicators, and styled UI.
  - Registered health commands in Tauri invoke handler.
- **Prophet Changepoint Detection (Phase 2)**:
  - Implemented automatic changepoint detection using PELT-inspired algorithm (`get_or_detect_changepoints()`).
  - Added CUSUM-based scoring for identifying significant structural changes (`compute_changepoint_score()`).
  - Implemented piecewise linear trend with changepoint matrix A(t) and rate adjustment deltas.
  - Added `solve_ridge_with_priors()` for different regularization on trend vs seasonal parameters.
- **Vectorized Environment for Parallel RL Training (Phase 2)**:
  - Created `VectorizedTradingEnv` class following Gymnasium VectorEnv interface.
  - Implemented `SubprocVecEnv` for true multi-process parallelism (bypasses GIL).
  - Added async step execution with `step_async()` and `step_wait()` methods.
  - Created `make_vec_env()` factory function for easy environment creation.
- **Logging & Visualization (P3 Phase 3)**:
  - Implemented **Logit Lens** for `NSTransformer` models in `visualize_utils.py`, allowing visualization of internal prediction evolution.
  - Modernized `visualize_utils.py` and `loss_landscape_workflow.py` to support sequential trading data and the `TradingEnv`.
  - Achieved full **Mypy strict mode** compliance across all visualization utilities.
- **Risk Management System (Phase 3)**:
  - Created `rust/src/simulation/risk.rs` with comprehensive risk monitoring.
  - Implemented `RiskManager` with position sizing limits, daily loss limits, and max drawdown.
  - Added historical VaR (Value at Risk) calculation using percentile method.
  - Implemented Sharpe and Sortino ratio calculations.
  - Automatic position reduction with `position_multiplier` on limit breaches.
  - 8 unit tests covering all risk scenarios.
- **Model Checkpoint Cloud Storage (Phase 3)**:
  - Created `python/src/utils/io/cloud_storage.py` with S3 and GCS backends.
  - Implemented `CloudCheckpointManager` for unified cloud model storage.
  - Added zstd compression for efficient checkpoint transfer.
  - Fallback to local storage on cloud failure.
  - Added optional `cloud` dependency group to `pyproject.toml`.
- **GPU Profiling & Optimization (Phase 3)**:
  - Created `python/src/utils/profiling/gpu_optimization.py` with optimization utilities.
  - Implemented `MemoryPool` for GPU memory pre-allocation.
  - Added `TransferProfiler` for Python↔Rust transfer profiling.
  - Added `GPUMemoryOptimizer` with bottleneck detection and memory estimation.
  - Added `optimize_for_inference()` with torch.compile support.
  - Added `get_gpu_optimization_recommendations()` for hardware-aware tips.

- **Foundation & Core Features Catch-up**:
  - **Streaming DataLoader**: Implemented `StreamingFinancialDataset` for memory-efficient loading of large CSV/Parquet files.
  - **Model Lifecycle**: Added `ModelRetentionPolicy` to automatically clean up old checkpoints based on count or metric.
  - **Secrets Management**: Integrated HashiCorp Vault with environment variable fallback in `SecretsManager`.
  - **Troubleshooting**: Created `TROUBLESHOOTING.md` guide.
  - **Concurrent Market Data**: Refactored Rust Polymarket scraper to fetch history concurrently (10x parallelization).

- **Testing & Quality (Priority 3)**:
  - **GPU Test Suite**: Added dedicated GPU tests for mixed precision and OOM handling.
  - **Performance CI**: Implemented automated benchmark regression detection (>10% threshold).
  - **Visual Testing**: Added Cypress visual regression tests for Dashboard and Terminal.
  - **Code Quality**: Enabled standard type checking strictness and added mutation testing configuration.

### Recent Updates (2026-01)

- **Phase 4: Optimization & Scale**:
  - Implemented Active Learning with uncertainty estimation (Quantile Regression, MC Dropout).
  - Integrated FinBERT sentiment analysis for financial news and social media.
  - Added intelligent sample selection (Entropy, BALD, Uncertainty samplers).
  - Created news crawler for RSS feeds (Yahoo Finance, MarketWatch, CNBC).
- **Rust Performance Optimizations**:
  - Added `SmallVec` for stack-allocated returns history (avoids heap for small windows).
  - Implemented pre-allocated `ObservationBuffer` for zero-allocation observation generation.
  - Verified with Criterion benchmarks: `trading_env_step` at 22.8µs, `reset` at 62.5ns.
- **Advanced ML: Ensemble Models**:
  - Implemented `EnsembleModel` wrapper with average, weighted, voting, and stacking strategies.
  - Added `predict_with_uncertainty` for ensemble disagreement-based uncertainty estimation.
  - Factory function `create_ensemble_from_configs` for easy ensemble creation.
- **Advanced ML: Meta-Learning**:
  - Implemented `MAMLWrapper` for Model-Agnostic Meta-Learning.
  - Created `RegimeDetector` for market regime classification (volatile, trending, ranging).
  - Enables rapid strategy adaptation when market conditions change.
- **Modular CLI Framework (P3 Phase 4)**:
  - Refactored the monolithic `command_parser.py` into a modular architecture.
  - Created a centralized registry (`registry.py`) and specialized parsers for `train`, `inference`, `webcrawler`, and `hp_optim`.
  - Consolidated HPO suite in `optimize.py` with support for Optuna, Ray Tune, and **DEHB**.
  - Integrated **Loguru** for structured and colorized logging during HPO trials.
  - Implemented multi-fidelity evaluation worker for efficient resource allocation in search.
  - Finalized PyTorch **DDP** integration with standalone CLI support.
  - Updated the modular CLI framework to include HPO and Distributed Training commands.
- **Distributed Training (Ray Tune Integration)**:
  - Integrated **Ray Tune** for hyperparameter optimization, leveraging PyTorch Lightning for training orchestration.
  - Set up **ASHA scheduler** and **Optuna search** for efficient search space exploration.
  - Implemented `ray_tune.py` with `RayTrainReportCallback` for seamless metric reporting.
- **API Documentation**:
  - Expanded **Rust** crate documentation with detailed module overviews, architecture descriptions, and usage examples.
  - Created a dedicated `rust/docs` directory for native doc storage.
  - Set up **Sphinx** for Python API documentation, including docstrings for all core models (`LVQ`, `RandomForest`, `SVM`, etc.).
  - Configured **TypeDoc** for TypeScript frontend documentation, covering critical hooks like `useArena`.
  - Generated HTML docs available in `rust/docs/`, `docs/` (Python), and `typescript/docs/` (TS).
- **DevOps & Containerization**:
  - Created a multi-stage **Dockerfile** for minimized production images (Rust + Python).
  - Added **docker-compose.yml** for orchestrated deployment including core API and monitoring (Prometheus/Grafana).
- **Performance Profiling & Optimization**:
  - Implemented continuous benchmarking via GitHub Actions (`cargo bench`).
  - Added a `@profile` decorator in `python/src/utils/profiling.py` for detailed Python function analysis.
  - Integrated profiling into the main training entry point.
- **Type Safety & Persistence (P2 Phase 1)**:
  - Created Python type stubs (`nglab/_nglab.pyi`) for the Rust module.
  - Enabled **strict Mypy checks** in `pyproject.toml` for the Python codebase.
  - Defined initial PostgreSQL schema for trades, portfolio snapshots, and model checkpoints.
  - Implemented SQLAlchemy 2.0+ models in `python/src/db/models.py`.
- **Observability & Versioning (P2 Phase 2)**:
  - Implemented Rust Prometheus metrics exporter for real-time performance tracking.
  - Created high-density **Grafana dashboard** for simulation metrics.
  - Integrated **MLflow** for robust model versioning and experiment tracking.
- **Model Serving & Backtesting (P3 Phase 2)**:
  - Implemented high-performance **FastAPI** inference service in `python/src/api/inference.py`.
  - Developed a modular **Backtesting Framework** in `python/src/backtesting/` wrapping the Rust `PolymarketArena`.
  - Integrated performance metrics (Sharpe, Sortino, Drawdown) for strategy evaluation.
  - Achieved full **Mypy strict mode** compliance for the new modules.
- **Enhanced Training Logging**:
  - Rebuilt `log_utils.py` with **WandB** integration, automated **Matplotlib** plotting, and JSON resilience.
  - Standardized training metrics logging in `train.py`, including loss, gradient norms, and representative predictions.
  - Achieved full **Mypy strict mode** compliance across the logging and training pipeline.
- **Documentation & Tracing (P2 Phase 3)**:
  - Established **Architecture Decision Records (ADR)** process in `docs/adr/`.
  - Integrated **OpenTelemetry** and **Jaeger** for distributed tracing in the Rust backend.
  - Created **"Getting Started"** interactive Jupyter notebook tutorial.
- **Health Checks**:
  - Implemented Flask-based health monitoring API in `python/src/api/health.py` (CPU, Memory, GPU).
  - Added `rust/src/health.rs` with serializable health status structures for Tauri integration.
- **Configuration Management**:
  - Added `config` crate for environment-based settings loading.
  - Created `config/development.toml`, `staging.toml`, and `production.toml` for tiered deployment configuration.
  - Implemented `Settings` struct in `rust/src/config.rs` for type-safe configuration access.
- **Production Logging**:
  - Implemented structured JSON logging using `tracing` and `tracing-appender` in `rust/src/logging.rs`.
  - Added daily log rotation to `logs/nglab.log`.
  - Added `ArenaError` with `thiserror` for comprehensive, strongly-typed error handling across the Rust backend.
  - Implemented `From<ArenaError> for PyErr` for seamless error propagation to Python.

### Changed
- **Gymnasium API**:
  - Refactored `TradingEnv` to return `PyResult` instead of panicking on errors.
  - Added `#[instrument]` tracing spans to `step` and `reset` methods for performance observability.

### Fixed
- **Rust Lints & Benchmarks**:
  - Suppressed `async_fn_in_trait` lint in `scraper.rs` for cleaner internal API.
  - Fixed compilation errors and deprecated usages in `rust/benchmarks/` (`arena_bench`, `orderbook_bench`, `trading_env_bench`), ensuring zero-warning baseline.

## [Unreleased] - 2026-01-19

### Added

- **Phase 2 Completion (Performance & Scaling)**:
  - **GPU Acceleration**: Implemented `GPUFeatureEngineer` using PyTorch for 100x speedup in technical indicator calculation (SMA, EMA, RSI, MACD, Bollinger Bands).
  - **Environment Batching**: Created `VectorizedTradingEnv` for running multiple simulation environments in parallel (Process/Thread pool).
  - **TorchRL Integration**: Updated `TradingEnvWrapper` to correctly handle `batch_size` and expose vectorized specs to TorchRL's `ParallelEnv`, enabling massive-scale RL training.
  - **Benchmarking**: Added `benchmark_scaling.py` to profile FPS scaling across CPU cores.
  - **Property Testing**: Integrated `hypothesis` for robust property-based testing of GPU features, verifying correctness against CPU baselines and numerical stability.

- **Phase 4 Completion (Scale & Polish)**:
  - **FastAPI Scaling**: Implemented `BatchInferenceHandler` and Redis caching layer in `inference.py` to support high-throughput async prediction serving. Added production Gunicorn config.
  - **Automated Pipeline**: Created `FeaturePipeline` combining `GPUFeatureEngineer`, `RobustScaler`, and `VarianceThreshold` for streamlined, automated feature engineering.

- **Online Learning (Phase 5 Foundation)**:
  - **Drift Detection**: Implemented `PageHinkley` and `MovingAverageDrift` detectors in `python/src/online/drift.py` for real-time concept drift monitoring.
  - **Online Normalization**: Created `OnlineNormalizer` (PyTorch module) using Welford's algorithm and momentum for adapting to shifting data distributions.
  - **Unit Tests**: Added comprehensive test suite `test_online_learning.py` verifying drift detection and normalization logic.


## [Unreleased] - 2026-01-18

### Added

- **Frontend Testing Suite**:
  - **Vitest Integration**: Setup Vitest with `@testing-library/react` and `jsdom` environment for component testing.
  - **Mocking Infrastructure**: Created robust mocks for Tauri APIs (`invoke`, `listen`), `ResizeObserver`, and `lightweight-charts` to enable isolated unit testing.
  - **Component Tests**:
    - **Dashboard**: Added tests for `GlobalActivityWidget`, `UserProfileWidget`, `OrderBook` (LOB visualization), `TrendingMarketsWidget`, and `DashboardOverview`.
    - **Terminal**: Added tests for `TerminalLayout`, `TerminalChart`, `OrderBookWidget`, `MarketSidebar`, `TradingFormWidget`, and `RecentTradesWidget`.
    - **Charts**: Added tests for `PriceChart` data updates and lifecycle.
    - **Tabs**: Added tests for all major application tabs: `ScraperTab`, `AnalysisTab`, `PredictionTab`, `TrainingTab`, `NewsTab`, `VaultTab`, `AccountTab`, `FavoritesTab`, `PricingTab`.
    - **App Logic**: Verified main `App` routing and global modal interactions.
    - **Hooks**: Added unit tests for custom hooks `useArena`, `usePolymarket`, and `useFavorites`.
  - **Coverage**: Achieved 100% pass rate across 28 test files and 115 test cases, providing full regression coverage for the UI layer.
- **Cypress E2E Test Suite**:
  - Added comprehensive Cypress end-to-end testing infrastructure for the TypeScript frontend.
  - Created 13 test spec files covering all major components: Navigation, Dashboard, Favorites, Login Modal, Simulation, Scraper, Pricing, Training, Vault, Account, Terminal/Markets, Prediction/Intelligence, and News tabs.
  - Implemented Tauri API mocking in `cypress/support/e2e.ts` with default responses for all backend commands (markets, vault, training, simulation, prediction, pricing).
  - Added custom Cypress commands (`mockTauriInvoke`, `navigateToTab`, `waitForMarketsToLoad`, `clearFavorites`) in `cypress/support/commands.ts`.
  - Created mock market data fixtures in `cypress/fixtures/markets.json`.
  - Added npm scripts for running tests: `cy:open`, `cy:run`, `cy:e2e`, `cy:e2e:open`.

### Changed

- **Terminal**:
  - **Favorites**: Integrated `useFavorites` hook into `TerminalLayout` and `MarketSidebar` for persistent favorite markets.
  - **Multi-Outcome**: Updated `TradingFormWidget` and `MarketSidebar` to support multi-outcome markets (e.g., electing Fed Chair) with dynamic outcome selection and pricing.
  - **UI**: Improved `TradingFormWidget` outcome toggle and button text for clarity.

- **Rust Backend & Bindings**:
  - **Refactoring**: Decoupled Python bindings from Rust core logic in `gym.rs`, `orderbook.rs`, `polymarket.rs`, and `lib.rs`, resolving PyO3 compilation errors and attribute clashes.
  - **Gymnasium API**: Updated `TradingEnv` to match Gymnasium standards (`reset` returns `(obs, info)` tuple, accepts `seed` and `options`).
  - **Features**: Exposed `load_prices` method in `TradingEnv` to enable injecting custom price data from Python.
  - **Testing**: Added `python/tests/test_gym_loop.py` for binding verification and updated `test_integration.py` to match new API signatures. Achieved 100% pass rate on integration tests.

## [Unreleased] - 2026-01-17

### Added

- **Comprehensive Documentation**:
  - Added high-fidelity JSDoc to the entire TypeScript frontend, including React components (Tabs, Dashboard, Terminal), custom hooks, and utility functions.
  - Converted Rust codebase documentation to standard inner-module style (`//!`) and added detailed item-level documentation for all structs, enums, traits, and functions in `rust/src/` and `rust/benchmarks/`.
  - Added module-level documentation to CSS files (`App.css`, `index.css`) to improve stylistic transparency.
  - Documented the Python Gymnasium environment wrappers in `environment/` using PEP 257 docstrings; verified compliance with `check_docstrings.py`.
  - Added top-level package initialization documentation to `nglab/__init__.py`.
  - Documented all utility scripts in `scripts/` with detailed headers for usage and maintenance.
- **Classical Machine Learning Models (Expanded)**:
  - **Regression**: Added LARS, Stepwise, M5, MARS, LOESS, and classical linear variants.
  - **Decision Trees**: Added comprehensive suite including CART, ID3, C4.5, C5.0, CHAID, DecisionStump, and ConditionalTree.
  - **Ensemble Methods**: Implemented AdaBoost, Bagging, Stacking, Voting, WeightedAverage (Blending), and GBRT.
  - **Bayesian**: Added GaussianNB, MultinomialNB, AODE, and BayesianNetwork (BBN).
  - **SVM Variants**: Added LinearSVM, NuSVM, OneClassSVM (Anomaly), LS-SVM, and Twin SVM (TWSVM).
  - **Clustering**: Added K-Medians (custom L1) and Expectation Maximisation (EM via GMM).
  - **Association Rules**: Added Eclat algorithm for vertical itemset mining.
  - **Association Rule Learning**: Added `EclatAlgorithm` (custom) alongside `Apriori` and `FPGrowth`.
  - **Dimensionality Reduction (Expanded)**: Added `PCR`, `PLSR`, `MDS`, `Sammon Mapping` (custom), `Projection Pursuit` (FastICA), `QDA`, `MDA` (custom), `FDA` (MARS-based), and `UMAP` (wrapper).
  - **Integration**: All models wrapped in `ClassicalModel` and integrated into `TimeSeriesBackbone` or `HelperModelFactory` for seamless PyTorch interoperability.
- **CI/CD & Code Quality**:
  - Integrated a comprehensive quality suite with `Black`, `Ruff`, `MyPy`, `Pip-Audit`, and `Pytest-Cov` for Python.
  - Re-integrated `cargo fmt` and `cargo clippy` for Rust quality assurance.
  - Added `Prettier` for TypeScript and JavaScript formatting with strict isolation.
  - Integrated `ktlint` via Gradle for the Android Kotlin codebase.
  - Implemented strict **Language Isolation** and directory exclusions in `pre-commit` to prevent unintentional changes to Markdown, Kotlin, and metadata files.
  - Standardized `Ruff` configuration to use the modern `[tool.ruff.lint]` structure and updated `MyPy` to Python 3.10 support.
  - Implemented `HelperModelFactory` for unified access to supplemental ML algorithms.
- **Code Organization**:
  - **Deep Models Refactoring**: Reorganized `python/src/models/deep/` into logical subdirectories (`autoencoders/`, `recurrent/`, `convolutional/`, `attention/`, `memory/`, `probabilistic/`, `general/`, `spiking/`, `competitive/`) with individual class files and facade modules for cleaner imports.
  - **MAC Models Refactoring**: Reorganized `python/src/models/mac/` into subdirectories (`linear/`, `trees/`, `boosting/`, `ensemble/`, `naive_bayes/`, `neighbors/`, `svm/`) with individual class files (46 total) and facade modules mirroring the `helper/` structure.
  - **Factory Pattern**: Split `time_series.py` into `deep_factory.py` and `mac_factory.py` to separate model creation logic, reducing the main file from 588 lines to ~70 lines while maintaining full backward compatibility.
- **Project Structure & Dependencies**:
  - Added `scikit-learn`, `xgboost`, and `lightgbm` to `pyproject.toml`.
  - Created modularized test fixtures in `python/tests/fixtures/` (`deep_fixtures.py`, `mac_fixtures.py`).
  - Added `ktlint` Gradle plugin support to the Android module.

### Changed

- **Tauri Backend Refactoring**:
  - Modularized the Tauri `lib.rs` into specialized submodules for state management (`state.rs`) and categorized command handlers (`commands/`).
- **Project Structure**: Improved maintainability of `typescript/src-tauri/src/` by decoupling commands from the main library entry point.
- **Test Infrastructure**: Updated `conftest.py` with global fixture loading and root path resolution for Python tests.

### Fixed

- **Build Environment**: Resolved linker issues in `.cargo/config.toml` to support `cargo doc` and standard builds in heterogeneous environments.
- **Deep Learning Models**: Fixed relative imports and crashing bugs in `nstansformer`, `tsmamba`, and `xlstm`.
- **Test Stability**: Resolved shape mismatches in `VAE` tests and `NameError` in classical model fitting unit tests.
- **Code Standards**: Fixed wildcard imports in Android Kotlin tests and adapters to comply with `ktlint` standards.
- **Python Quality**: Addressed various import errors and shape mismatches identified during the CI integration of classical and deep models.

## [Unreleased] - 2026-01-16

### Added

- **Neural Network Architectures (Advanced)**:
  - **Perceptron (P)**: Basic single-layer feedforward network with configurable activations.
  - **Markov Chain (MC)**: Probabilistic state transition model with learnable matrices.
  - **Boltzmann Machine (BM)**: Stochastic recurrent network with symmetric connections and energy-based learning.
  - **Deep Belief Network (DBN)**: Stack of RBMs with greedy layer-wise training.
  - **Deep Convolutional Network (DCN)**: Hierarchical CNN with BN and pooling for deep features.
  - **Deconvolutional Network (DN)**: Transposed convolutions for upsampling and reconstruction.
  - **Deep Convolutional Inverse Graphics Network (DCIGN)**: Disentangled representation learning (pose, lighting vs identity).
  - **Liquid State Machine (LSM)**: Reservoir computing with spiking neurons and fixed sparse recurrence.
  - **Deep Residual Network (DRN)**: Residual blocks with skip connections for very deep training.
  - **Differentiable Neural Computer (DNC)**: External addressable memory with content/temporal linkage.
  - **Neural Turing Machine (NTM)**: Addressable external memory with shift/sharpening mechanisms.
  - **Attention Network (AN)**: Multi-head self-attention mechanism with positional encoding.
  - **Normalizing Flow (Flow)**: RealNVP-based generative model with invertible affine coupling layers.
  - **Neural ODE (NODE)**: Continuous-time depth model with RK4 solver.
  - **Physics-Informed Neural Network (PINN)**: MLP with gradient supervision for PDE solving.
  - **Stacked Auto-Encoders (SAE)**: Deep AutoEncoder composed of stacked shallow AEs.
- **Neural Network Architectures (Standard)**:
  - Implemented Spiking Neural Network (SNN) with custom `LIFCell`.
  - Added MLP, RBF, AE, DAE, SAE, Hopfield Network, ESN, ELM, KohonenMap (SOM), and Capsule Layers.
  - Implemented Rolling Window CNN, TimeGAN, and Diffusion U-Net (1D).
- **Integration**:
  - Fully integrated all new models into the `TimeSeriesBackbone` factory.
  - Added support for `output_type` ('prediction' vs 'embedding') across all backbone models.
  - Added `return_sequence` support for all applicable architectures.
- **Frontend / Dashboard**:
  - **News Tab**: New tab for aggregating news feeds from customized sources (Crypto, Social, Market Data).
  - **Training Tab**: Dedicated interface for configuring and training neural network models directly from the UI.
  - **Prediction Tab**: Added "Deep Learning" model selection to run inference with pre-trained PyTorch models.
  - **Dashboard UI**: Refined `UserProfileWidget` with "Profile Stats" design and improved PnL charts (dynamic coloring, sparkline style).
  - **Navigation**: optimizing tab ordering for better workflow (News moved to end).
- **Backend (Rust & Python)**:
  - **Commands**: Added `list_trained_models` and `predict_trained_model` Tauri commands.
  - **Inference**: Created `infer.py` for standalone model inference via subprocess.
  - **Refactoring**: Reorganized deep learning models into `python/src/models/deep/` for better structure.
- **CI/CD**:
  - Created GitHub Actions workflow for automated Python, Rust, and TypeScript testing/linting.

### Fixed

- Standardized RNN (LSTM/GRU) and xLSTM interfaces for backbone compatibility.
- Fixed VAE KL-annealing logic and reconstruction shape mismatches.
- Resolved various import and type-check issues in the Python pipeline.

### Changed

- **Modularity**: Split the model library into individual specialized files for better maintainability.
- **Testing**: Consolidated architecture tests into a structured, class-based suite in `test_architectures.py`.
- Updated `walkthrough.md` with comprehensive documentation for all new architectures.
