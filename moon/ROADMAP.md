# NGLab Roadmap

<a href="https://www.gnu.org/licenses/agpl-3.0"><img alt="License: AGPL v3" src="https://img.shields.io/badge/License-AGPL_v3-blue.svg"></a>

This document is the **master roadmap** for NGLab. It covers cross-cutting product features; the
per-module engineering roadmaps live under [`moon/roadmaps/`](roadmaps/).

Completed items are moved to [`docs/CHANGELOG.md`](../docs/CHANGELOG.md).

---

## Module Roadmaps

NGLab is a **polyglot trading architecture**. Each tier has a dedicated roadmap:

| Roadmap | Tier / Language | Scope |
| :--- | :--- | :--- |
| [Crypto Daemon — Go](roadmaps/crypto_go.md) | Tier 2 / Warm Path (**Go**) | Exchange WebSocket feeds, JSON-RPC nodes, concurrent crypto trading, loopback IPC to Rust |
| [HFT Native Loop — C++](roadmaps/hft_cpp.md) | Tier 1 / Hot Path (**C++**) | Sub-microsecond execution, DOD order-book matching, shared-memory IPC |
| [Core Hub — Rust](roadmaps/core_rust.md) | Tier 0 / Control (**Rust**) | Tauri backend, prediction markets, EVM/Alloy monitoring, binary lifecycle management |
| [Strategy Brain — Python](roadmaps/strategy_python.md) | Offline / Analytical (**Python**) | AI/ML models, quant strategies, prediction-weight export |
| [Control Panel — TypeScript](roadmaps/frontend_typescript.md) | UI (**TypeScript**) | Thin consumer of data streams + execution triggers |
| [Universal Schema — Protobuf](roadmaps/schema_protobuf.md) | Cross-boundary | `Order`/`Position`/`Tick` schemas, codegen for TS/Rust/Go/C++ |
| [Code Quality & Human Understanding](roadmaps/code_quality.md) | Cross-cutting | Docs, naming, architecture cleanup, DX, testing, coverage targets |

See [`ARCHITECTURE.md`](../docs/ARCHITECTURE.md) for the tier diagram and IPC boundaries.

---

## Polyglot Migration (Priority 0)

The highest-priority track: extract Crypto and HFT logic out of Rust into the tiers where they
belong, per the migration directives.

| # | Item | Direction | Roadmap |
| :--- | :--- | :--- | :--- |
| M.1 | Migrate crypto trading logic, exchange WS feeds, JSON-RPC node connections | Rust → **Go** | [crypto_go.md](roadmaps/crypto_go.md) |
| M.2 | Go loopback server on `127.0.0.1`, dynamic `--port` handshake from Rust | New (Go) | [crypto_go.md §2](roadmaps/crypto_go.md) |
| M.3 | Migrate sub-µs execution loops, raw order-book matching, Tier-1 logic | Rust → **C++** | [hft_cpp.md](roadmaps/hft_cpp.md) |
| M.4 | C++ shared-memory (`shm_open`/`mmap`) metrics bridge (zero-copy) | New (C++) | [hft_cpp.md §2](roadmaps/hft_cpp.md) |
| M.5 | Protobuf universal schema + codegen for TS/Rust/Go/C++ | New | [schema_protobuf.md](roadmaps/schema_protobuf.md) |
| M.6 | Rust lifecycle manager spins up/monitors/restarts Go + C++ binaries | Rust | [core_rust.md](roadmaps/core_rust.md) |

**Enforcement:** all future crypto/exchange/concurrent-feed work is **Go**; all future
ultra-low-latency/HFT-venue work is **C++**; prediction markets + EVM stay **native Rust**.

---

## Executive Summary

NGLab has a strong foundation with a Rust simulation engine, Python ML layer, and Tauri/React frontend. This plan focuses on expanding trading capabilities, enhancing the user interface, and adding new analytical features.

**Focus Areas:**

1. **Advanced Trading Features** - New order types, execution algorithms, market mechanics
2. **Enhanced Visualization** - Charts, analytics, real-time dashboards
3. **Portfolio & Strategy Tools** - Backtesting UI, strategy builder, portfolio analytics
4. **Data Integration** - New data sources, streaming improvements
5. **ML Pipeline UI** - Training visualization, model management
6. **Simulation Enhancements** - Market microstructure, multi-agent systems

---

## Priority 1: Advanced Trading Features (High Impact)

### 1.1 Extended Order Types

**Files**: `rust/src/simulation/orderbook.rs`, `typescript/src-tauri/src/lib.rs`
**Status**: Basic order types exist (limit, market, stop, iceberg, trailing)
**Impact**: High - Professional trading capabilities

```
Tasks:
[x] Implement Fill-or-Kill (FOK) orders
  - Execute entire order immediately or cancel
  - Add FOK flag to Order struct
[x] Implement Immediate-or-Cancel (IOC) orders
  - Execute what's available, cancel remainder
  - Partial fill tracking
[x] Add Good-Till-Date (GTD) orders
  - Expiration timestamp field
  - Background cleanup task for expired orders
[x] Implement bracket orders
  - Entry + stop-loss + take-profit as atomic unit
  - Automatic child order creation on fill
[x] Add pegged orders
  - Peg to mid, bid, ask, or last price
  - Dynamic price adjustment on book updates
[x] Expose new order types via Tauri commands
[x] Add order type selector in TradingFormWidget
```

**Complexity**: Medium | **Impact**: High

### 1.2 Algorithmic Execution Engine

**Files**: `rust/src/simulation/multi_asset.rs`, new `rust/src/execution/`
**Status**: Basic TWAP/VWAP exist in multi-asset env
**Impact**: High - Institutional-grade execution

```
Tasks:
[x] Create dedicated execution module
  - rust/src/execution/mod.rs
  - rust/src/execution/twap.rs
  - rust/src/execution/vwap.rs
  - rust/src/execution/pov.rs (Percentage of Volume)
[x] Implement TWAP with randomization
  - Configurable time slices
  - Random jitter to avoid detection
  - Progress tracking and cancellation
[x] Implement adaptive VWAP
  - Historical volume profile integration
  - Real-time volume participation adjustment
[x] Add Percentage of Volume (POV) algorithm
  - Target participation rate
  - Reaction to market volume
[x] Add Implementation Shortfall algorithm
  - Minimize execution cost vs. arrival price
  - Urgency parameter for trade-off
[x] Create execution analytics
  - Slippage measurement
  - Market impact estimation
  - Execution quality reports
[x] Add algorithm selector in Terminal UI
[x] Real-time execution progress visualization
```

**Complexity**: High | **Impact**: High

### 1.3 Market Maker Mode

**Files**: new `rust/src/simulation/market_maker.rs`
**Status**: Not implemented
**Impact**: Medium - Strategy diversification

```
Tasks:
[x] Implement spread quoting engine
  - Configurable bid-ask spread
  - Position-based skew adjustment
  - Inventory risk management
[x] Add quote refresh logic
  - Time-based requoting
  - Event-driven updates (trade, book change)
[x] Implement adverse selection protection
  - Cancel quotes on large market orders
  - Asymmetric spread widening
[x] Create P&L tracking for market making
  - Realized spread capture
  - Inventory costs
  - Rebate tracking
[x] Add market maker dashboard widget
[x] Configuration panel for MM parameters
```

**Complexity**: High | **Impact**: Medium

### 1.4 Multi-Leg Order Support

- [x] Define multi-leg order structures (Spread, Butterfly, Calendar) [rust/src/simulation/spreads.rs]
- [x] Integrate spread execution logic into `MultiAssetEnv` [rust/src/simulation/multi_asset.rs]
      **Status**: Single-leg orders only
      **Impact**: Medium - Options and spread trading

```
Tasks:
[x] Implement spread order type
  - Two-leg simultaneous execution
  - Net price specification
[x] Add butterfly spread support
  - Three-leg atomic execution
  - Ratio specification
[x] Implement calendar spread orders
  - Same asset, different expiries
  - Roll mechanics
[x] Create combo order builder UI
  - Visual leg configuration
  - Net payoff diagram
[x] Add spread order book visualization
[x] Implied pricing calculations
```

**Complexity**: High | **Impact**: Medium

---

## Priority 2: Enhanced Visualization (High Impact)

### 2.1 Advanced Charting Features

**Files**: `typescript/src/components/charts/`, `typescript/src/components/terminal/`
**Status**: Basic candlestick and line charts via lightweight-charts
**Impact**: High - Professional analysis tools

```
Tasks:
[x] Add multiple chart types
  - Heikin-Ashi candles
  - Renko charts
  - Point & Figure
  - Kagi charts
[x] Implement drawing tools
  - Trend lines with persistence
  - Fibonacci retracements
  - Support/resistance zones
  - Text annotations
[x] Add technical indicators overlay
  - Moving averages (SMA, EMA, WMA)
  - Bollinger Bands
  - MACD histogram
  - RSI with overbought/oversold zones
  - Volume profile
[x] Create multi-timeframe view
  - Synchronized crosshair across charts
  - Timeframe selector (1m, 5m, 15m, 1h, 4h, 1d)
[x] Implement chart templates
  - Save/load indicator configurations
  - Preset templates for common setups
[x] Add chart comparison mode
  - Overlay multiple assets
  - Correlation visualization
```

**Complexity**: Medium | **Impact**: High

### 2.2 Order Book Visualization Enhancements

**Files**: `typescript/src/components/OrderBook.tsx`, `typescript/src/components/terminal/OrderBookWidget.tsx`
**Status**: Basic top 15 levels display
**Impact**: Medium - Market depth understanding

```
Tasks:
[x] Implement depth chart visualization
  - Cumulative bid/ask curves
  - Interactive hover for price levels
  - Zoom and pan controls
[x] Add order book heatmap
  - Color intensity by size
  - Historical depth comparison
[x] Implement order flow imbalance indicator
  - Real-time bid/ask pressure
  - Divergence alerts
[x] Add trade tape visualization
  - Time & Sales with size coloring
  - Aggressor side indication
  - Large trade highlighting
[x] Create order book replay
  - Historical snapshots playback
  - Speed control
[x] Add iceberg detection indicators
  - Hidden liquidity estimation
  - Reload pattern recognition
```

**Complexity**: Medium | **Impact**: Medium

### 2.3 Real-Time Analytics Dashboard

**Files**: new `typescript/src/components/analytics/`
**Status**: Basic risk dashboard exists
**Impact**: High - Actionable insights

```
Tasks:
[x] Create performance attribution widget
  - P&L by asset
  - P&L by strategy
  - Time-based breakdown (hourly, daily)
[x] Implement trade analytics panel
  - Win rate visualization
  - Average win vs. average loss
  - Profit factor trending
[x] Add position monitoring grid
  - All positions with real-time P&L
  - Unrealized vs. realized
  - Position aging
[x] Create correlation matrix heatmap
  - Asset correlation visualization
  - Rolling correlation
[x] Implement drawdown visualization
  - Underwater equity curve
  - Recovery time tracking
[x] Add volatility dashboard
  - Historical vs. implied vol
  - Vol surface visualization (3D)
  - Term structure chart
```

**Complexity**: Medium | **Impact**: High

### 2.4 Notification & Alert System

**Files**: new `typescript/src/components/notifications/`, `typescript/src-tauri/src/notifications.rs`
**Status**: No alert system
**Impact**: Medium - Active monitoring

```
Tasks:
[x] Implement price alert system
  - Price crosses level
  - Percentage change threshold
  - Volume spike detection
[x] Add technical indicator alerts
  - RSI overbought/oversold
  - MA crossover
  - Bollinger Band breach
[x] Create risk alerts
  - Drawdown threshold breach
  - Position limit warnings
  - VaR limit approach
[x] Implement notification center UI
  - Alert history log
  - Snooze/dismiss functionality
  - Priority levels
[x] Add system tray notifications (Tauri)
[x] Create alert sound configuration
[x] Implement alert persistence (SQLite)
```

**Complexity**: Medium | **Impact**: Medium

---

## Priority 3: Portfolio & Strategy Tools (High Impact)

### 3.1 Visual Backtesting Interface

**Files**: new `typescript/src/components/backtesting/`, `typescript/src-tauri/src/backtesting.rs`
**Status**: Python backtesting engine exists, no UI
**Impact**: High - Strategy development workflow

```
Tasks:
[x] Create backtest configuration panel
  - Date range selector
  - Initial capital input
  - Transaction cost settings
  - Slippage model selection
[x] Implement strategy selector
  - List available strategies from Python
  - Parameter configuration forms
  - Strategy code preview
[x] Add backtest execution controls
  - Start/pause/stop buttons
  - Progress bar with ETA
  - Real-time equity curve update
[x] Create results dashboard
  - Performance metrics summary
  - Equity curve chart
  - Drawdown chart
  - Monthly returns heatmap
[x] Implement trade log viewer
  - Filterable trade list
  - Trade markers on chart
  - Individual trade analysis
[x] Add backtest comparison view
  - Side-by-side metrics
  - Overlaid equity curves
  - Statistical significance tests
[x] Create backtest report export (PDF/CSV)
```

**Complexity**: High | **Impact**: High

### 3.2 Strategy Builder (No-Code)

**Files**: new `typescript/src/components/strategy-builder/`
**Status**: Not implemented
**Impact**: High - Accessibility for non-programmers

```
Tasks:
[x] Create visual rule builder
  - Drag-and-drop conditions
  - IF-THEN-ELSE logic blocks
  - Indicator condition nodes
[x] Implement condition types
  - Price conditions (above, below, crosses)
  - Indicator conditions (RSI, MA, etc.)
  - Time conditions (market hours, day of week)
  - Position conditions (has position, P&L threshold)
[x] Add action blocks
  - Market/limit order actions
  - Position sizing rules
  - Stop-loss/take-profit attachment
[x] Create strategy validation
  - Syntax checking
  - Logic contradiction detection
  - Backtest preview
[x] Implement strategy code generation
  - Export to Python strategy class
  - Readable code output
[x] Add strategy templates library
  - Common strategies (MA crossover, breakout)
  - User-saved templates
```

**Complexity**: High | **Impact**: High

### 3.3 Portfolio Optimization UI

**Files**: new `typescript/src/components/portfolio/`, integrate with `python/src/models/portfolio_optimizer.py`
**Status**: Python optimizer exists, no UI
**Impact**: Medium - Asset allocation tools

```
Tasks:
[x] Create efficient frontier visualization
  - Risk-return scatter plot
  - Frontier curve
  - Interactive point selection
[x] Implement asset weight sliders (In PortfolioAllocation component)
  - Manual weight adjustment
  - Constraint visualization
  - Real-time metrics update
[x] Add optimization parameter panel
  - Target return/risk inputs
  - Constraint configuration (min/max weights)
  - Rebalancing frequency
[x] Create correlation analysis view
  - Asset correlation matrix
  - Diversification score
  - Concentration risk metrics
[x] Implement rebalancing suggestions
  - Current vs. target weights
  - Trade list generation
  - Transaction cost estimation
[x] Add historical optimization analysis (Planned)
  - Rolling efficient frontier
  - Regime analysis
```

**Complexity**: Medium | **Impact**: Medium

### 3.4 Paper Trading Mode

**Files**: `typescript/src-tauri/src/lib.rs`, new state management
**Status**: Implemented
**Impact**: Medium - Risk-free practice

```
Tasks:
[x] Create paper trading account system
  - Virtual balance tracking
  - Separate from simulation
  - Persistent across sessions
[x] Implement realistic execution simulation
  - Configurable fill rates
  - Slippage modeling
  - Partial fill simulation
[x] Add paper trading indicator in UI
  - Clear visual distinction from live
[x] Create paper trading performance tracking
  - Separate P&L history
[x] Implement paper-to-live transition checks
  - Validated account state on startup
  - Reset and toggle logic
[ ] Add paper trading leaderboard (optional)
```

**Complexity**: Medium | **Impact**: Medium

---

## Priority 4: Data Integration (Medium Impact)

### 4.1 Multi-Exchange Support

**Files**: new `rust/src/web/exchanges/`, `typescript/src-tauri/src/exchanges.rs`
**Status**: Polymarket only
**Impact**: High - Broader market coverage

```
Tasks:
[x] Create exchange abstraction layer
  - Common interface for all exchanges
  - Unified order types mapping
  - Normalized market data format
[x] Implement Binance integration
  - [x] REST API for data fetching
  - [x] WebSocket for real-time updates
  - [x] Order submission (paper mode)
[x] Implement Kraken integration
  - Spot and futures support
  - OHLCV data fetching
[x] Implement Deribit integration
  - Options data
  - Perpetual futures
[x] Create exchange selector in UI
  - [x] Connection status indicators
  - [x] API key management per exchange
[x] Add cross-exchange arbitrage view
  - [x] Price comparison
  - [x] Spread monitoring
```

**Complexity**: High | **Impact**: High

### 4.2 Alternative Data Sources

**Files**: new `rust/src/web/alternative/`, `python/src/data/`
**Status**: Price data only
**Impact**: Medium - Enhanced signals

```
Tasks:
[x] Implement news feed integration
  - RSS aggregation
  - Keyword filtering
  - Sentiment tagging
[x] Add social sentiment data
  - Twitter/X API integration (placeholder)
  - Reddit sentiment scraping (placeholder)
  - Aggregated sentiment score
□ Create on-chain data integration
  - Whale wallet tracking
  - Exchange flow monitoring
  - Network metrics (active addresses)
[x] Implement economic calendar
  - Event fetching
  - Impact classification
  - Countdown timers
□ Add data source quality metrics
  - Latency monitoring
  - Data completeness scores
[x] Create alternative data dashboard (AlternativeDataTab)
```

**Complexity**: High | **Impact**: Medium

### 4.3 Historical Data Management

**Files**: `typescript/src/components/data-manager/`, `typescript/src-tauri/src/commands/datasets.rs`
**Status**: Implemented with Data Catalog and Storage Stats
**Impact**: Medium - Data organization

```
Tasks:
[x] Create data catalog UI
  - [x] Available datasets listing
  - [x] Metadata (size, last modified)
  - [x] Dataset search and filtering
[x] Implement dataset management
  - [x] Delete dataset from disk
[x] Add dataset preview tool
  - [x] List available columns/features
[x] Add storage management
  - [x] Disk usage visualization
  - [x] Total dataset count and size stats
[ ] Implement data quality tools
  - [x] Placeholder for gap detection
  - [ ] Outlier identification
[ ] Create data export functionality
  - [x] CSV export placeholder
  - [ ] Parquet export
[ ] Implement data versioning
  - [ ] Track data updates
```

**Complexity**: Medium | **Impact**: Medium

### 4.4 Real-Time Data Streaming Improvements

**Files**: `rust/src/web/streaming.rs`, `typescript/src-tauri/src/lib.rs`
**Status**: Basic WebSocket streaming
**Impact**: Medium - Reliability and features

```
Tasks:
[x] Implement reconnection logic
  - [x] Exponential backoff
  - [x] State recovery after reconnect
  - [x] Gap detection and fill (via state recovery)
[ ] Add data compression
  - [ ] Delta compression for orderbook
  - [ ] Configurable compression levels
[x] Create stream health monitoring
  - [x] Latency tracking
  - [x] Message rate monitoring
  - [x] Drop detection
□ Implement multi-stream aggregation
  - Combine multiple sources
  - Best bid/ask aggregation
□ Add stream recording
  - Record raw stream to file
  - Playback capability
□ Create bandwidth optimization
  - Subscription management
  - Level-of-detail control
```

**Complexity**: Medium | **Impact**: Medium

---

## Priority 5: ML Pipeline UI (Medium Impact)

### 5.1 Training Dashboard

**Files**: new `typescript/src/components/training/`, enhance `typescript/src-tauri/src/lib.rs`
**Status**: Basic train_model command exists
**Impact**: High - ML workflow visibility

```
Tasks:
[x] Create training job queue UI
  - Job listing with status
  - Priority management
  - Cancel/pause capability
[x] Implement real-time training metrics
  - Loss curves (train/val)
  - Learning rate schedule
  - Gradient statistics
[x] Add hyperparameter visualization
  - Current hyperparameters display
  - Historical comparison
[x] Create GPU utilization monitor
  - Memory usage
  - Compute utilization
  - Temperature monitoring
[x] Implement training logs viewer
  - Filterable log stream
  - Error highlighting
  - Export capability
[x] Add early stopping controls
  - Manual stop with checkpoint
  - Patience configuration
[x] Create training history browser
  - Past experiments listing
  - Metrics comparison
```

**Complexity**: Medium | **Impact**: High

### 5.2 Model Registry UI

**Files**: new `typescript/src/components/models/`
**Status**: Implemented - Catalog, Comparison, Documentation, Filtering
**Impact**: Medium - Model management

```
Tasks:
[x] Create model catalog interface
  - [x] Model listing with metadata
  - [ ] Version history per model
  - [ ] Performance metrics summary
[x] Implement model comparison view
  - [x] Side-by-side metrics
  - [ ] Architecture differences
  - [ ] Training data differences
[x] Add model deployment controls
  - [x] Set active inference model
  - [ ] A/B testing configuration
  - [ ] Rollback capability
[x] Create model documentation panel
  - [x] Auto-generated model card
  - [x] Training configuration
  - [x] Input/output specifications
[x] Implement model search and filter
  - [x] By architecture type
  - [ ] By performance metrics
  - [ ] By creation date
[ ] Add model export/import
  - [ ] ONNX export
  - [ ] Model sharing
```

**Complexity**: Medium | **Impact**: Medium

### 5.3 Feature Engineering UI

**Files**: new `typescript/src/components/features/`
**Status**: Implemented - Catalog with Statistics, Feature Builder, Validation Tools
**Impact**: Medium - Feature development workflow

```
Tasks:
[x] Create feature catalog
  - [x] Available features listing
  - [x] Feature statistics (mean, std, min, max)
  - [x] Correlation with target
[x] Implement feature builder
  - [x] Visual feature combination
  - [x] Mathematical operations
  - [x] Lag/lead transformations
[x] Add feature importance visualization
  - [x] SHAP values display
  - [ ] Permutation importance
  - [x] Feature ranking
[x] Create feature validation tools
  - [x] Distribution analysis (histogram)
  - [x] Stationarity tests (ADF)
  - [ ] Look-ahead bias detection
[ ] Implement feature set management
  - Save/load feature sets
  - Version control
[ ] Add feature documentation
  - Auto-generated descriptions
  - Usage statistics
```

**Complexity**: Medium | **Impact**: Medium

### 5.4 Prediction Explanation UI

**Files**: new `typescript/src/components/explanations/`
**Status**: Implemented - Feature Importance, Attention Heatmaps, SHAP, Prediction Confidence
**Impact**: Medium - Model interpretability

```
Tasks:
[x] Create prediction breakdown view
  - [x] Feature contribution waterfall/bar chart
  - [x] Top contributing factors (SHAP importance)
  - [x] Confidence intervals
[x] Implement SHAP visualization
  - [x] Summary plots
  - [ ] Dependence plots
  - [ ] Force plots
[ ] Add counterfactual explanations
  - [ ] "What-if" analysis
  - [ ] Minimum change for different outcome
[x] Create attention visualization
  - [x] For transformer models
  - [x] Temporal attention heatmap
[x] Implement prediction confidence display
  - [x] Uncertainty quantification
  - [x] Out-of-distribution detection
□ Add explanation export
  - PDF reports
  - API for programmatic access
```

**Complexity**: High | **Impact**: Medium

---

## Priority 6: Simulation Enhancements (Medium Impact)

### 6.1 Market Microstructure Features

**Files**: `rust/src/simulation/orderbook.rs`, new modules
**Status**: Basic CLOB mechanics
**Impact**: Medium - Realistic simulation

```
Tasks:
[x] Implement auction mechanisms
  - Opening auction simulation
  - Closing auction
  - Volatility auctions
[x] Add market maker simulation
  - Automated liquidity provision
  - Spread dynamics
  - Inventory management
[x] Create latency simulation
  - Configurable order latency
  - Market data delay
  - Jitter modeling
[x] Implement queue position tracking
  - Accurate fill simulation
  - Queue priority visualization
[x] Add tick size rules
  - Price increment enforcement
  - Lot size rules
[x] Create circuit breaker simulation
  - Trading halts
  - Price limits
```

**Complexity**: High | **Impact**: Medium

### 6.2 Multi-Agent Simulation

**Files**: new `rust/src/simulation/multi_agent.rs`
**Status**: Single agent only
**Impact**: Medium - Market dynamics

```
Tasks:
[x] Create agent framework
  - Agent trait definition
  - Agent lifecycle management
  - Inter-agent messaging
[x] Implement `LOBFeatureGenerator` [Task 4.2.1]
  - Define features: imbalance, spread, VWAP, derivatives
  - Integrated into Gym step through `FeaturePipeline`
[x] Implement Recursive Feature Elimination (RFE) [Task 4.2.2]
  - Complete RFE implementation in `feature_selection.py`
  - Integrated into `FeaturePipeline`
[x] Implement `MarketRegimeDetector` [Task 4.2.3]
  - GMM-based clustering of market states
  - Integrated as one-hot input to policy
[x] Add automated scaling using `OnlineNormalizer` [Task 4.2.4]
  - Welford's algorithm for online mean/std calculation
  - Integrated into `FeaturePipeline` as 'online' scaler
[x] Implement agent types
  - Momentum traders
  - Mean reversion traders
  - Market makers
  - Noise traders
[x] Add agent parameterization
  - Configurable aggression
  - Capital allocation
  - Strategy parameters
[x] Create market impact visualization
  - Agent activity heatmap
  - Price impact attribution
[x] Implement agent performance tracking
  - Per-agent P&L
  - Market share metrics
[x] Add scenario builder
  - Configure agent population
  - Event injection
```

**Complexity**: High | **Impact**: Medium

### 6.3 Options Simulation

**Files**: new `rust/src/simulation/options.rs`, integrate with `rust/src/models/black_scholes.rs`
**Status**: Pricing models exist, no options trading simulation
**Impact**: Medium - Derivatives trading

```
Tasks:
[x] Implement options order book
  - Separate books per strike/expiry
  - Options-specific order types
[x] Add Greeks calculation engine
  - Real-time Greeks updates
  - Portfolio-level Greeks
[x] Create options chain UI
  - Strike/expiry matrix
  - Bid/ask/last/volume
  - IV display
[x] Implement exercise/assignment
  - American vs. European
  - Early exercise logic
  - Assignment simulation
[x] Add volatility surface visualization
  - 3D surface plot
  - Smile/skew analysis
[x] Create options strategy builder
  - Multi-leg strategy construction
  - Payoff diagram
  - Break-even analysis
```

**Complexity**: High | **Impact**: Medium

### 6.4 Scenario Analysis Engine

**Files**: new `rust/src/simulation/scenarios.rs`, `typescript/src/components/scenarios/`
**Status**: Not implemented
**Impact**: Medium - Risk analysis

```
Tasks:
[x] Create scenario definition system
  - Price shock scenarios
  - Volatility spike scenarios
  - Liquidity crisis scenarios
[x] Implement stress testing
  - Portfolio stress test execution
  - Multiple scenario comparison
[x] Add historical scenario replay
  - Flash crash replay
  - Major event replay
[x] Create Monte Carlo simulation
  - Path generation
  - VaR/CVaR calculation
  - Distribution visualization
[x] Implement scenario builder UI
  - Visual scenario configuration

---

## Priority 7: Deep Learning Architectures (High Impact)

### 7.1 Advanced Model Library

**Files**: `python/src/models/deep/`
**Status**: Implemented
**Impact**: High - State-of-the-art forecasting capabilities

```

Tasks:
[x] Implement Competitive Networks (LVQ, SOM)
[x] Implement Convolutional Networks (CNN, ResNet, DCIGN, Capsule)
[x] Implement Memory Augmented Networks (DNC, NTM)
[x] Implement Probabilistic Models (TimeGAN, Diffusion, Flow, RBM)
[x] Implement Recurrent Networks (xLSTM, TSMamba, ESN, LSM)
[x] Implement Spiking Neural Networks (SNN, LIF)
[x] Implement Graph Neural Networks (GCN, GatedGCN)
[x] Create modular building blocks (Attention, Mamba, Embeddings)

```
**Complexity**: High | **Impact**: High
  - Parameter sliders
[x] Add scenario results dashboard
  - P&L distribution
  - Risk metrics under stress
```

**Complexity**: High | **Impact**: Medium

---

## Implementation Roadmap

### Phase A: Core Trading Features (Weeks 1-4)

- [x] Extended order types (1.1)
- [x] Advanced charting features (2.1)
- [x] Order book visualization (2.2)
- [x] Real-time analytics dashboard (2.3)

### Phase B: Strategy Tools (Weeks 5-8)

- [x] Visual backtesting interface (3.1)
- [x] Training dashboard (5.1)
- [x] Algorithmic execution engine (1.2)
- [ ] Notification system (2.4) (Partially implemented)

### Phase C: Data & Integration (Weeks 9-12)

- [x] Multi-exchange support (4.1)
- [x] Model registry UI (5.2)
- [x] Historical data management (4.3)
- [x] Paper trading mode (3.4)

### Phase D: Advanced Features (Weeks 13-16)

- [ ] Strategy builder (3.2)
- [ ] Market microstructure features (6.1)
- [ ] Options simulation (6.3)
- [ ] Multi-agent simulation (6.2)

### Phase E: Polish & Extensions (Weeks 17-20)

- [ ] Portfolio optimization UI (3.3)
- [ ] Alternative data sources (4.2)
- [ ] Feature engineering UI (5.3)
- [ ] Scenario analysis engine (6.4)

---

## Quick Wins (< 1 day each)

These improvements can be implemented quickly with high value:

1. **Add Heikin-Ashi chart type** - Simple transformation of existing candle data
2. **Implement price alerts** - Basic threshold checking on price updates
3. **Add trade tape to terminal** - Display recent trades from orderbook
4. **[x] Create position P&L column** - Real-time unrealized P&L calculation
5. **Add timeframe selector** - Switch between candle intervals
6. **Implement chart crosshair** - Coordinated cursor across charts
7. **Add model list dropdown** - Simple UI for model selection
8. **Create backtest date picker** - Date range input component
9. **[x] Add export to CSV button** - Download trade history (Implemented for Positions)
10. **Implement dark/light theme toggle** - CSS variable switching

---

## Feature Dependencies

```
Extended Order Types (1.1)
    └── Algo Execution (1.2)
         └── Market Maker Mode (1.3)

Advanced Charting (2.1)
    └── Backtest Visualization (3.1)
         └── Strategy Builder (3.2)

Training Dashboard (5.1)
    └── Model Registry (5.2)
         └── Prediction Explanation (5.4)

Multi-Exchange (4.1)
    └── Alternative Data (4.2)
         └── Scenario Analysis (6.4)
```

---

## Metrics for Success

| Feature Area     | Metric               | Target          |
| ---------------- | -------------------- | --------------- |
| Order Types      | Types supported      | 10+             |
| Charts           | Indicator count      | 20+             |
| Backtesting      | Avg backtest time    | <30s for 1yr    |
| Data Sources     | Exchanges integrated | 4+              |
| ML UI            | Training visibility  | Full pipeline   |
| Simulation       | Agents supported     | 100+ concurrent |
| UI Response      | Chart render time    | <16ms (60fps)   |
| Strategy Builder | Rule types           | 15+             |

---

## File Structure for New Features

```
typescript/src/components/
├── analytics/
│   ├── PerformanceAttribution.tsx
│   ├── TradeAnalytics.tsx
│   ├── CorrelationMatrix.tsx
│   └── DrawdownChart.tsx
├── backtesting/
│   ├── BacktestConfig.tsx
│   ├── BacktestResults.tsx
│   ├── TradeLog.tsx
│   └── BacktestComparison.tsx
├── strategy-builder/
│   ├── RuleBuilder.tsx
│   ├── ConditionNode.tsx
│   ├── ActionNode.tsx
│   └── StrategyCanvas.tsx
├── training/
│   ├── TrainingDashboard.tsx
│   ├── MetricsChart.tsx
│   ├── HyperparameterView.tsx
│   └── TrainingLogs.tsx
├── models/
│   ├── ModelCatalog.tsx
│   ├── ModelComparison.tsx
│   └── ModelCard.tsx
├── notifications/
│   ├── AlertCenter.tsx
│   ├── AlertConfig.tsx
│   └── NotificationToast.tsx
└── scenarios/
    ├── ScenarioBuilder.tsx
    ├── StressTest.tsx
    └── MonteCarloResults.tsx

rust/src/
├── execution/
│   ├── mod.rs
│   ├── twap.rs
│   ├── vwap.rs
│   └── implementation_shortfall.rs
├── simulation/
│   ├── market_maker.rs (new)
│   ├── multi_agent.rs (new)
│   ├── options.rs (new)
│   └── scenarios.rs (new)
└── web/
    └── exchanges/
        ├── mod.rs
        ├── binance.rs
        ├── kraken.rs
        └── deribit.rs
```

---

## Priority 7: Python Codebase Architecture (Based on WSmart-Route/logic)

This section outlines architectural improvements for the Python codebase, derived from patterns and best practices observed in WSmart-Route/logic.

### 7.1 Directory Structure Reorganization

**Current State**: Scattered constants, implicit CLI, deep model hierarchy
**Target State**: Domain-organized structure following WSmart patterns

```
Current Structure Issues:
python/src/
├── models/deep/          # 10+ nested subdirectories
├── env/                  # Generic, unclear structure
├── configs/              # Hydra-heavy, complex
└── (no constants/)       # Constants scattered throughout

Proposed Structure:
python/src/
├── cli/                  # NEW: Explicit CLI module
│   ├── __init__.py
│   ├── commands.py       # Subcommand routing
│   ├── args.py           # Argument definitions
│   └── validators.py     # Input validation
├── constants/            # NEW: Domain-organized constants
│   ├── __init__.py
│   ├── models.py         # Model-related constants
│   ├── paths.py          # File path templates
│   ├── training.py       # Training hyperparameters
│   ├── trading.py        # Trading-specific constants
│   ├── system.py         # System-wide constants
│   └── testing.py        # Test configuration
├── configs/              # Simplified dataclasses-first
│   ├── __init__.py
│   ├── base.py           # Base config classes
│   ├── model.py          # Model configurations
│   ├── training.py       # Training configurations
│   └── environment.py    # Environment configurations
├── envs/                 # RENAME: env/ → envs/ with clear hierarchy
│   ├── __init__.py
│   ├── base.py           # RL4COEnvBase equivalent
│   ├── trading.py        # Trading environment
│   ├── polymarket.py     # Polymarket-specific
│   └── generators.py     # Data generators
├── models/               # FLATTEN: 2-3 level max hierarchy
│   ├── modules/          # Atomic components
│   ├── subnets/          # Composed encoder/decoder units
│   ├── policies/         # Policy networks
│   └── registry.py       # Central MODEL_REGISTRY
└── ...
```

```
Tasks:
[x] Create python/src/cli/ module
  - Move CLI logic from main.py
  - Implement subcommand routing (train, evaluate, backtest)
  - Add argument validators
[x] Create python/src/constants/ with domain files
  - Extract constants from configs into dedicated files
  - Organize by: models, paths, training, trading, system, testing
[x] Flatten models/deep/ hierarchy
  - Reorganize into modules/, subnets/, policies/
  - Remove deep nesting (max 2-3 levels)
[x] Rename env/ to envs/ with clear inheritance
  - Create base.py with abstract environment class
  - Implement proper ABC hierarchy
[x] Update all imports after restructure
[x] Add __all__ exports to all __init__.py files
```

**Complexity**: High | **Impact**: High

---

### 7.2 Factory Pattern Implementation

**Current State**: Direct instantiation, minimal factories
**Target State**: Abstract factories with component registries (WSmart pattern)

```python
# TARGET: Factory interface pattern from WSmart

# python/src/models/factories/base.py
from abc import ABC, abstractmethod
from typing import Any
import torch.nn as nn

class NeuralComponentFactory(ABC):
    """Abstract factory for creating neural network components."""

    @abstractmethod
    def create_encoder(self, **kwargs: Any) -> nn.Module:
        """Create encoder module."""
        pass

    @abstractmethod
    def create_decoder(self, **kwargs: Any) -> nn.Module:
        """Create decoder module."""
        pass

    @abstractmethod
    def create_embedding(self, **kwargs: Any) -> nn.Module:
        """Create embedding layer."""
        pass

# python/src/models/factories/attention.py
class AttentionComponentFactory(NeuralComponentFactory):
    """Factory for attention-based components."""

    def create_encoder(self, **kwargs: Any) -> nn.Module:
        from ..subnets.attention_encoder import AttentionEncoder
        return AttentionEncoder(**kwargs)

    def create_decoder(self, **kwargs: Any) -> nn.Module:
        from ..subnets.attention_decoder import AttentionDecoder
        return AttentionDecoder(**kwargs)

# python/src/policies/factory.py
class PolicyFactory:
    """Factory for creating policy instances."""

    @staticmethod
    def get_policy(policy_name: str, **kwargs) -> "IPolicy":
        if 'neural' in policy_name.lower():
            return NeuralPolicy(**kwargs)
        elif 'rule_based' in policy_name:
            return RuleBasedPolicy(**kwargs)
        elif 'ensemble' in policy_name:
            return EnsemblePolicy(**kwargs)
        else:
            raise ValueError(
                f"Unknown policy: {policy_name}. "
                f"Available: {list(POLICY_REGISTRY.keys())}"
            )
```

```
Tasks:
[x] Create python/src/models/factories/ module
  - Implement NeuralComponentFactory ABC
  - Create concrete factories: AttentionFactory, ConvolutionalFactory, RecurrentFactory
[x] Create python/src/policies/factory.py
  - Implement PolicyFactory with pattern matching
  - Add clear error messages with available options
[x] Create python/src/envs/factory.py
  - Implement environment factory
  - Registry pattern for all environments
[x] Create python/src/pipeline/factory.py
  - Pipeline component factory
  - Trainer factory with algorithm selection
[x] Add component registries
  - MODEL_REGISTRY, POLICY_REGISTRY, ENV_REGISTRY, PIPELINE_REGISTRY
  - Registration decorators
[x] Update all instantiation sites to use factories
```

**Complexity**: High | **Impact**: High

---

### 7.3 Abstract Base Classes & Protocols

**Current State**: Limited ABC usage, basic Policy ABC
**Target State**: Comprehensive ABC hierarchy (WSmart pattern)

```python
# TARGET: Expanded ABC pattern from WSmart

# python/src/envs/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional
import torch

if TYPE_CHECKING:
    from tensordict import TensorDict

class TradingEnvBase(ABC):
    """Unified trading environment interface."""

    name: str = "base"

    @property
    def batch_size(self) -> torch.Size:
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value: torch.Size) -> None:
        if not isinstance(value, torch.Size):
            value = torch.Size(value if isinstance(value, (list, tuple)) else [value])
        self._batch_size = value

    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> TensorDict:
        """Reset environment and return initial state."""
        pass

    @abstractmethod
    def step(self, action: TensorDict) -> TensorDict:
        """Execute action and return next state."""
        pass

    @abstractmethod
    def get_reward(self, td: TensorDict) -> torch.Tensor:
        """Calculate reward from state."""
        pass

# python/src/models/policies/base.py
class ConstructivePolicy(nn.Module, ABC):
    """Base class for constructive policies with multiple inheritance."""

    @abstractmethod
    def forward(
        self,
        td: TensorDict,
        env: TradingEnvBase,
        decode_type: str = "sampling",
        **kwargs
    ) -> dict:
        """Forward pass for policy."""
        pass

    @abstractmethod
    def evaluate(self, td: TensorDict) -> dict:
        """Evaluate policy without exploration."""
        pass
```

```
Tasks:
[x] Create comprehensive TradingEnvBase in python/src/envs/base.py
  - Add property decorators with validation
  - Implement error handling in setters
  - Define all required abstract methods
[x] Expand Policy ABC in python/src/policies/base.py
  - Add ConstructivePolicy for neural models
  - Add ImprovementPolicy for iterative refinement
  - Define evaluation protocol
[x] Create model ABCs in python/src/models/base.py
  - BaseEncoder, BaseDecoder, BaseEmbedding
  - Forward method signatures with type hints
[x] Create pipeline ABCs in python/src/pipeline/base.py
  - BaseTrainer, BaseEvaluator, BaseCallback
[x] Add Protocol classes for duck typing where appropriate
[x] Update all concrete classes to inherit from ABCs
```

**Complexity**: Medium | **Impact**: High

---

### 7.4 Type Hints & Import Patterns

**Current State**: Mixed type hint usage, inconsistent imports
**Target State**: Comprehensive typing with TYPE_CHECKING pattern

```python
# TARGET: WSmart typing pattern

# EVERY module should start with:
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

if TYPE_CHECKING:
    # Imports only needed for type hints (avoids circular imports)
    from python.src.envs.base import TradingEnvBase
    from python.src.models.policies.base import ConstructivePolicy
    from tensordict import TensorDict
    import torch

# Function signatures with comprehensive types
def train_model(
    model: nn.Module,
    env: TradingEnvBase,
    config: TrainConfig,
    callbacks: Optional[List[Callback]] = None,
    checkpoint_path: Optional[str] = None,
) -> Tuple[nn.Module, Dict[str, float]]:
    """
    Train a model in the given environment.

    Args:
        model: The neural network model to train.
        env: The trading environment instance.
        config: Training configuration dataclass.
        callbacks: Optional list of training callbacks.
        checkpoint_path: Optional path to save checkpoints.

    Returns:
        Tuple of (trained_model, metrics_dict).

    Raises:
        ValueError: If config is invalid.
        RuntimeError: If training fails.
    """
    ...
```

```
Tasks:
[x] Add `from __future__ import annotations` to ALL modules
[x] Implement TYPE_CHECKING pattern for circular imports
  - Identify all circular import issues
  - Move type-only imports under TYPE_CHECKING
[x] Add comprehensive type hints to all functions
  - Return types
  - Parameter types
  - Optional vs required distinction
[x] Create type stubs for complex protocols
  - python/src/types.pyi or inline stubs
[x] Add comprehensive docstrings with Args/Returns/Raises
  - Follow Google-style docstring format
  - Document all parameters
[x] Configure mypy for strict type checking
  - Add mypy.ini or pyproject.toml config
  - Fix all type errors
[x] Add type: ignore comments only where absolutely necessary
```

**Complexity**: Medium | **Impact**: Medium

---

### 7.5 Configuration Management Simplification

**Current State**: Heavy Hydra dependency, MISSING sentinels
**Target State**: Dataclasses-first with optional YAML override

```python
# TARGET: WSmart-style dataclass configs

# python/src/configs/base.py
from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class EnvConfig:
    """Environment configuration."""
    name: str = "trading"
    num_envs: int = 1
    max_steps: int = 1000
    device: str = "cuda"

@dataclass
class ModelConfig:
    """Model configuration."""
    architecture: str = "transformer"
    embedding_dim: int = 128
    hidden_dim: int = 256
    num_heads: int = 8
    num_layers: int = 6
    dropout: float = 0.1

@dataclass
class TrainConfig:
    """Training configuration."""
    algorithm: str = "ppo"
    learning_rate: float = 3e-4
    batch_size: int = 64
    num_epochs: int = 100
    gradient_clip: float = 1.0
    seed: int = 42

@dataclass
class Config:
    """Root configuration with nested configs."""
    env: EnvConfig = field(default_factory=EnvConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load config from YAML file with overrides."""
        import yaml
        with open(path) as f:
            overrides = yaml.safe_load(f)
        return cls._apply_overrides(overrides)

    @classmethod
    def _apply_overrides(cls, overrides: dict) -> "Config":
        """Apply dictionary overrides to default config."""
        # Implementation...
```

```
Tasks:
[x] Refactor configs to pure dataclasses
  - Remove MISSING sentinels
  - Add sensible defaults for all fields
  - Nested composition with field(default_factory=...)
[x] Create Config.from_yaml() method
  - Optional YAML override loading
  - Validate loaded values
[x] Implement deep_sanitize() utility
  - Convert DictConfig/ListConfig to primitives
  - Use before passing to Lightning modules
[x] Remove Hydra decorator from main.py
  - Use explicit config loading
  - Keep CLI argument parsing separate
[x] Create config validation layer
  - Validate ranges and constraints
  - Clear error messages for invalid configs
[x] Add config serialization/deserialization
  - to_dict(), from_dict() methods
  - JSON/YAML export
```

**Complexity**: Medium | **Impact**: Medium

---

### 7.6 Error Handling & Validation

**Current State**: Basic exceptions, minimal validation
**Target State**: Explicit error handling with context (WSmart pattern)

```python
# TARGET: WSmart error handling pattern

# python/src/exceptions.py
class NGLabError(Exception):
    """Base exception for NGLab."""
    pass

class ConfigurationError(NGLabError):
    """Raised when configuration is invalid."""
    pass

class ModelNotFoundError(NGLabError):
    """Raised when a model cannot be found."""
    pass

class EnvironmentError(NGLabError):
    """Raised when environment encounters an error."""
    pass

# python/src/envs/base.py
@batch_size.setter
def batch_size(self, value: torch.Size) -> None:
    """Set batch size with validation and error handling."""
    if not isinstance(value, torch.Size):
        if isinstance(value, int):
            value = torch.Size([value])
        elif isinstance(value, (list, tuple)):
            value = torch.Size(value)
        else:
            raise TypeError(
                f"batch_size must be torch.Size, int, list, or tuple. "
                f"Got: {type(value).__name__}"
            )

    if any(v <= 0 for v in value):
        raise ValueError(
            f"batch_size must contain positive values. Got: {value}"
        )

    try:
        self._batch_size = value
        self._sync_spec_shapes()
    except RuntimeError as e:
        raise EnvironmentError(
            f"Failed to set batch_size to {value}: {e}"
        ) from e

# python/src/models/registry.py
def get_model(name: str) -> type:
    """Get model class by name."""
    if name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise ModelNotFoundError(
            f"Unknown model: '{name}'. Available models: {available}"
        )
    return MODEL_REGISTRY[name]
```

```
Tasks:
[x] Create python/src/exceptions.py with custom exceptions
  - NGLabError base class
  - Domain-specific exceptions (Config, Model, Env, Training)
[x] Add validation decorators
  - @validate_config, @validate_input
  - Reusable validation patterns
[x] Implement explicit error handling in property setters
  - Type validation
  - Range validation
  - Graceful fallbacks where appropriate
[x] Add context to all error messages
  - What went wrong
  - What was expected
  - Available alternatives (for registry lookups)
[x] Create error logging utilities
  - Structured error logging
  - Stack trace preservation
[x] Add input validation at system boundaries
  - User input validation
  - External API response validation
```

**Complexity**: Medium | **Impact**: Medium

---

### 7.7 Test Organization Enhancement

**Current State**: 7 fixture modules, flat test structure
**Target State**: 12+ fixture modules, organized by domain (WSmart pattern)

```
Current:
python/tests/
├── conftest.py
├── fixtures/
│   ├── model_fixtures.py
│   ├── nglab_fixtures.py
│   └── ... (7 files)
└── test_*.py (flat)

Target:
python/tests/
├── conftest.py                 # Central pytest config with plugin registration
├── fixtures/
│   ├── __init__.py
│   ├── arg_fixtures.py         # CLI argument fixtures
│   ├── config_fixtures.py      # Configuration fixtures
│   ├── data_fixtures.py        # Data/dataset fixtures
│   ├── env_fixtures.py         # Environment fixtures
│   ├── model_fixtures.py       # Model fixtures
│   ├── policy_fixtures.py      # Policy fixtures
│   ├── pipeline_fixtures.py    # Training pipeline fixtures
│   ├── integration_fixtures.py # Integration test fixtures
│   ├── mock_fixtures.py        # Mock objects
│   └── tensor_fixtures.py      # Tensor/tensor dict fixtures
├── unit/
│   ├── test_configs.py
│   ├── test_models/
│   │   ├── test_encoders.py
│   │   ├── test_decoders.py
│   │   └── test_policies.py
│   ├── test_envs/
│   └── test_utils/
├── integration/
│   ├── test_training_loop.py
│   ├── test_env_integration.py
│   └── test_pipeline.py
└── properties/
    └── test_model_properties.py
```

```python
# TARGET: conftest.py with plugin registration (WSmart pattern)

# python/tests/conftest.py
import pytest

pytest_plugins = [
    "tests.fixtures.arg_fixtures",
    "tests.fixtures.config_fixtures",
    "tests.fixtures.data_fixtures",
    "tests.fixtures.env_fixtures",
    "tests.fixtures.model_fixtures",
    "tests.fixtures.policy_fixtures",
    "tests.fixtures.pipeline_fixtures",
    "tests.fixtures.integration_fixtures",
    "tests.fixtures.mock_fixtures",
    "tests.fixtures.tensor_fixtures",
]

@pytest.fixture
def temp_output_dir():
    """Create temporary directory for test outputs."""
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir

@pytest.fixture
def device():
    """Get appropriate device for tests."""
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"
```

```
Tasks:
[x] Reorganize tests/ into unit/, integration/, properties/
[x] Expand fixtures/ to 10+ domain-specific modules
  - Add arg_fixtures.py, config_fixtures.py
  - Add tensor_fixtures.py for TensorDict fixtures
  - Add pipeline_fixtures.py
[x] Update conftest.py with pytest_plugins registration
[x] Add parametrized tests for factory methods
  - Test all registry entries
  - Test error cases
[x] Create test utilities module
  - Common assertions
  - Test data generators
[x] Add coverage configuration
  - Minimum coverage thresholds
  - Coverage report generation
[x] Implement property-based tests with hypothesis
  - Model invariant testing
  - Config validation testing
```

**Complexity**: Medium | **Impact**: Medium

---

### 7.8 Documentation Enhancement

**Current State**: Basic CLAUDE.md, minimal module docstrings
**Target State**: Comprehensive documentation (WSmart pattern - 400+ line CLAUDE.md)

```
Tasks:
□ Expand CLAUDE.md significantly
  - Add detailed section on codebase architecture
  - Document all major modules and their responsibilities
  - Add code patterns and conventions
  - Include troubleshooting section
  - Document config sanitization patterns
□ Create ARCHITECTURE.md
  - System diagrams (ASCII or Mermaid)
  - Data flow documentation
  - Component interaction descriptions
□ Create CONTRIBUTING.md
  - Code style guidelines
  - PR process
  - Testing requirements
□ Add comprehensive module docstrings
  - Every __init__.py should have module-level docstring
  - Explain module purpose and contents
□ Add inline documentation for complex logic
  - Algorithm explanations
  - Non-obvious design decisions
□ Create API documentation
  - Auto-generated from docstrings
  - Usage examples
```

**Complexity**: Medium | **Impact**: High

---

### 7.9 Package Initialization Patterns

**Current State**: Mixed star imports, inconsistent **all**
**Target State**: Explicit imports with registries (WSmart pattern)

```python
# TARGET: WSmart __init__.py pattern

# python/src/models/__init__.py
"""
Neural network models for NGLab trading system.

This module provides:
- MODEL_REGISTRY: Central registry of all model classes
- get_model(): Factory function to instantiate models by name
- Base classes for custom model implementation
"""
from __future__ import annotations

from python.src.models.base import BaseModel, BaseEncoder, BaseDecoder
from python.src.models.registry import MODEL_REGISTRY, get_model, register_model

# Import concrete implementations to trigger registration
from python.src.models import attention
from python.src.models import convolutional
from python.src.models import recurrent

__all__ = [
    # Base classes
    "BaseModel",
    "BaseEncoder",
    "BaseDecoder",
    # Registry
    "MODEL_REGISTRY",
    "get_model",
    "register_model",
]

# python/src/envs/__init__.py
"""
Trading environments for reinforcement learning.

Provides gym-compatible environments for training trading agents.
"""
from __future__ import annotations

from python.src.envs.base import TradingEnvBase
from python.src.envs.trading import TradingEnv
from python.src.envs.polymarket import PolymarketEnv

ENV_REGISTRY = {
    "trading": TradingEnv,
    "polymarket": PolymarketEnv,
}

def get_env(name: str, **kwargs) -> TradingEnvBase:
    """
    Get environment instance by name.

    Args:
        name: Environment name (trading, polymarket).
        **kwargs: Environment configuration.

    Returns:
        Configured environment instance.

    Raises:
        ValueError: If environment name is unknown.
    """
    if name not in ENV_REGISTRY:
        available = ", ".join(sorted(ENV_REGISTRY.keys()))
        raise ValueError(f"Unknown environment: '{name}'. Available: {available}")
    return ENV_REGISTRY[name](**kwargs)

__all__ = [
    "TradingEnvBase",
    "TradingEnv",
    "PolymarketEnv",
    "ENV_REGISTRY",
    "get_env",
]
```

```
Tasks:
□ Remove all star imports (from x import *)
□ Add explicit __all__ to every __init__.py
□ Create central registries in each major module
  - MODEL_REGISTRY, ENV_REGISTRY, POLICY_REGISTRY
□ Implement get_*() factory functions
  - get_model(), get_env(), get_policy()
□ Add registration decorators
  - @register_model, @register_env
□ Add module-level docstrings to all packages
□ Ensure consistent import ordering
  - Standard library
  - Third-party
  - Local imports
```

**Complexity**: Medium | **Impact**: Medium

---

### 7.10 Lightning Configuration Sanitization

**Current State**: Direct OmegaConf usage with Lightning
**Target State**: Sanitized configs before Lightning (WSmart critical pattern)

```python
# TARGET: Config sanitization utility (WSmart CLAUDE.md critical pattern)

# python/src/utils/config.py
from typing import Any, Dict
from omegaconf import DictConfig, ListConfig

def deep_sanitize(cfg: DictConfig | Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert DictConfig/ListConfig to primitive Python types.

    CRITICAL: Always use this before passing config to PyTorch Lightning
    modules to avoid YAML serialization errors.

    Args:
        cfg: Configuration object (DictConfig or dict).

    Returns:
        Dict with only primitive Python types.

    Example:
        >>> from omegaconf import OmegaConf
        >>> cfg = OmegaConf.create({"lr": 0.001, "layers": [64, 128]})
        >>> sanitized = deep_sanitize(cfg)
        >>> type(sanitized["layers"])
        <class 'list'>  # Not ListConfig
    """
    if isinstance(cfg, DictConfig):
        return {k: deep_sanitize(v) for k, v in cfg.items()}
    elif isinstance(cfg, ListConfig):
        return [deep_sanitize(v) for v in cfg]
    elif isinstance(cfg, dict):
        return {k: deep_sanitize(v) for k, v in cfg.items()}
    elif isinstance(cfg, list):
        return [deep_sanitize(v) for v in cfg]
    else:
        return cfg

# USAGE PATTERN:
# common_kwargs = deep_sanitize(cfg.model)
# common_kwargs["env"] = env  # Inject non-serializable AFTER sanitization
# model = MyLightningModule(**common_kwargs)
```

```
Tasks:
[x] Create python/src/utils/config.py with deep_sanitize()
[x] Audit all Lightning module instantiations
  - Find all places where configs are passed to Lightning
  - Add deep_sanitize() calls
□ Create helper for config injection pattern
  - sanitize_and_inject() utility
  - Standard pattern for env/model injection
□ Add tests for config sanitization
  - Test nested configs
  - Test with actual Lightning modules
□ Document in CLAUDE.md
  - Add section on config sanitization
  - Examples of correct usage
```

**Complexity**: Low | **Impact**: High

---

## Python Architecture Implementation Roadmap

### Phase 1: Foundation (HIGH IMPACT)

[x] Create `python/src/cli/` module (7.1)
[x] Create `python/src/constants/` with domain files (7.1)
[x] Implement deep_sanitize() utility (7.10)
[x] Create custom exceptions (7.6)

### Phase 2: Abstraction Layer (HIGH IMPACT)

[x] Implement factory patterns (7.2)
[x] Create comprehensive ABCs (7.3)
[x] Add component registries (7.9)
[x] Flatten model hierarchy (7.1)

### Phase 3: Type Safety & Quality (MEDIUM IMPACT)

- [x] Add `from __future__ import annotations` everywhere (7.4)
- [x] Implement TYPE_CHECKING pattern (7.4)
- [x] Refactor configs to pure dataclasses (7.5) [x]
- [x] Add explicit **all** exports (7.9)

### Phase 4: Testing & Documentation (MEDIUM IMPACT)

- [x] Reorganize test structure (7.7)
- [x] Expand to 10+ fixture modules (7.7)
- [x] Expand CLAUDE.md (7.8)
- [x] Create/Update ARCHITECTURE.md (7.8)

### Phase 5: Performance & Reliability (MEDIUM IMPACT)

- [x] Implement validation decorators and reusable patterns (7.6)
- [x] Add performance profiling hooks to CLI (7.10) [x]
- [x] Implement property-based testing with Hypothesis (7.7) [x]
- [x] Add memory usage tracking to Trainer (7.10) [x]
- [x] Create specialized mocking fixtures for Rust objects (7.7) [x]

---

## Quick Wins - Python Architecture

1. **Add `from __future__ import annotations`** - Single line per file, enables forward references
2. **[x] Create deep_sanitize() utility** - Prevents Lightning YAML errors
3. **[x] Add **all** to major **init**.py** - Explicit API surface
4. **[x] Create python/src/exceptions.py** - Centralized error types
5. **[x] Move constants to dedicated module** - Better organization
6. **[x] Add ENV_REGISTRY dict** - Simple factory lookup
7. **Create temp_output_dir fixture** - Common test pattern
8. **Add pytest_plugins to conftest.py** - Cleaner fixture loading
9. **[x] Add module docstrings** - Self-documenting packages
10. **[x] Remove star imports** - Explicit is better than implicit

---

## Summary: Key Improvements from WSmart-Route/logic

| Pattern               | Current Python | WSmart Target                 | Priority |
| --------------------- | -------------- | ----------------------------- | -------- |
| Directory Structure   | Scattered      | Domain-organized              | HIGH     |
| Factory Patterns      | Minimal        | ABC + Factories               | HIGH     |
| Abstract Base Classes | Limited        | Comprehensive                 | HIGH     |
| Type Hints            | Partial        | Universal                     | MEDIUM   |
| Constants             | Ad-hoc         | Dedicated module              | HIGH     |
| Configuration         | Hydra-heavy    | Dataclasses-first             | MEDIUM   |
| Error Handling        | Basic          | Explicit/defensive            | MEDIUM   |
| Test Organization     | Adequate       | 12+ fixture modules           | MEDIUM   |
| Documentation         | Minimal        | 400+ line CLAUDE.md           | HIGH     |
| Package Init          | Star imports   | Explicit **all** + registries | MEDIUM   |
| Config Sanitization   | None           | deep_sanitize()               | HIGH     |

---

_Last Updated: 2026-01-25 21:55_
_Version: 5.2 (Phase 5: Performance & Reliability - Completed)_
