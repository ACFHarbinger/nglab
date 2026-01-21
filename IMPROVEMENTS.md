# NGLab Feature Improvement Plan

This document outlines feature improvements for NGLab, focusing exclusively on backend and frontend capabilities. Each section includes actionable items with complexity and impact ratings.

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
□ Implement Fill-or-Kill (FOK) orders
  - Execute entire order immediately or cancel
  - Add FOK flag to Order struct
□ Implement Immediate-or-Cancel (IOC) orders
  - Execute what's available, cancel remainder
  - Partial fill tracking
□ Add Good-Till-Date (GTD) orders
  - Expiration timestamp field
  - Background cleanup task for expired orders
□ Implement bracket orders
  - Entry + stop-loss + take-profit as atomic unit
  - Automatic child order creation on fill
□ Add pegged orders
  - Peg to mid, bid, ask, or last price
  - Dynamic price adjustment on book updates
□ Expose new order types via Tauri commands
□ Add order type selector in TradingFormWidget
```
**Complexity**: Medium | **Impact**: High

### 1.2 Algorithmic Execution Engine

**Files**: `rust/src/simulation/multi_asset.rs`, new `rust/src/execution/`
**Status**: Basic TWAP/VWAP exist in multi-asset env
**Impact**: High - Institutional-grade execution

```
Tasks:
□ Create dedicated execution module
  - rust/src/execution/mod.rs
  - rust/src/execution/twap.rs
  - rust/src/execution/vwap.rs
  - rust/src/execution/pov.rs (Percentage of Volume)
□ Implement TWAP with randomization
  - Configurable time slices
  - Random jitter to avoid detection
  - Progress tracking and cancellation
□ Implement adaptive VWAP
  - Historical volume profile integration
  - Real-time volume participation adjustment
□ Add Implementation Shortfall algorithm
  - Minimize execution cost vs. arrival price
  - Urgency parameter for trade-off
□ Create execution analytics
  - Slippage measurement
  - Market impact estimation
  - Execution quality reports
□ Add algorithm selector in Terminal UI
□ Real-time execution progress visualization
```
**Complexity**: High | **Impact**: High

### 1.3 Market Maker Mode

**Files**: new `rust/src/simulation/market_maker.rs`
**Status**: Not implemented
**Impact**: Medium - Strategy diversification

```
Tasks:
□ Implement spread quoting engine
  - Configurable bid-ask spread
  - Position-based skew adjustment
  - Inventory risk management
□ Add quote refresh logic
  - Time-based requoting
  - Event-driven updates (trade, book change)
□ Implement adverse selection protection
  - Cancel quotes on large market orders
  - Asymmetric spread widening
□ Create P&L tracking for market making
  - Realized spread capture
  - Inventory costs
  - Rebate tracking
□ Add market maker dashboard widget
□ Configuration panel for MM parameters
```
**Complexity**: High | **Impact**: Medium

### 1.4 Multi-Leg Order Support

**Files**: `rust/src/simulation/orderbook.rs`, new types
**Status**: Single-leg orders only
**Impact**: Medium - Options and spread trading

```
Tasks:
□ Implement spread order type
  - Two-leg simultaneous execution
  - Net price specification
□ Add butterfly spread support
  - Three-leg atomic execution
  - Ratio specification
□ Implement calendar spread orders
  - Same asset, different expiries
  - Roll mechanics
□ Create combo order builder UI
  - Visual leg configuration
  - Net payoff diagram
□ Add spread order book visualization
□ Implied pricing calculations
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
□ Add multiple chart types
  - Heikin-Ashi candles
  - Renko charts
  - Point & Figure
  - Kagi charts
□ Implement drawing tools
  - Trend lines with persistence
  - Fibonacci retracements
  - Support/resistance zones
  - Text annotations
□ Add technical indicators overlay
  - Moving averages (SMA, EMA, WMA)
  - Bollinger Bands
  - MACD histogram
  - RSI with overbought/oversold zones
  - Volume profile
□ Create multi-timeframe view
  - Synchronized crosshair across charts
  - Timeframe selector (1m, 5m, 15m, 1h, 4h, 1d)
□ Implement chart templates
  - Save/load indicator configurations
  - Preset templates for common setups
□ Add chart comparison mode
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
□ Implement depth chart visualization
  - Cumulative bid/ask curves
  - Interactive hover for price levels
  - Zoom and pan controls
□ Add order book heatmap
  - Color intensity by size
  - Historical depth comparison
□ Implement order flow imbalance indicator
  - Real-time bid/ask pressure
  - Divergence alerts
□ Add trade tape visualization
  - Time & Sales with size coloring
  - Aggressor side indication
  - Large trade highlighting
□ Create order book replay
  - Historical snapshots playback
  - Speed control
□ Add iceberg detection indicators
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
□ Create performance attribution widget
  - P&L by asset
  - P&L by strategy
  - Time-based breakdown (hourly, daily)
□ Implement trade analytics panel
  - Win rate visualization
  - Average win vs. average loss
  - Profit factor trending
□ Add position monitoring grid
  - All positions with real-time P&L
  - Unrealized vs. realized
  - Position aging
□ Create correlation matrix heatmap
  - Asset correlation visualization
  - Rolling correlation
□ Implement drawdown visualization
  - Underwater equity curve
  - Recovery time tracking
□ Add volatility dashboard
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
□ Implement price alert system
  - Price crosses level
  - Percentage change threshold
  - Volume spike detection
□ Add technical indicator alerts
  - RSI overbought/oversold
  - MA crossover
  - Bollinger Band breach
□ Create risk alerts
  - Drawdown threshold breach
  - Position limit warnings
  - VaR limit approach
□ Implement notification center UI
  - Alert history log
  - Snooze/dismiss functionality
  - Priority levels
□ Add system tray notifications (Tauri)
□ Create alert sound configuration
□ Implement alert persistence (SQLite)
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
□ Create backtest configuration panel
  - Date range selector
  - Initial capital input
  - Transaction cost settings
  - Slippage model selection
□ Implement strategy selector
  - List available strategies from Python
  - Parameter configuration forms
  - Strategy code preview
□ Add backtest execution controls
  - Start/pause/stop buttons
  - Progress bar with ETA
  - Real-time equity curve update
□ Create results dashboard
  - Performance metrics summary
  - Equity curve chart
  - Drawdown chart
  - Monthly returns heatmap
□ Implement trade log viewer
  - Filterable trade list
  - Trade markers on chart
  - Individual trade analysis
□ Add backtest comparison view
  - Side-by-side metrics
  - Overlaid equity curves
  - Statistical significance tests
□ Create backtest report export (PDF/CSV)
```
**Complexity**: High | **Impact**: High

### 3.2 Strategy Builder (No-Code)

**Files**: new `typescript/src/components/strategy-builder/`
**Status**: Not implemented
**Impact**: High - Accessibility for non-programmers

```
Tasks:
□ Create visual rule builder
  - Drag-and-drop conditions
  - IF-THEN-ELSE logic blocks
  - Indicator condition nodes
□ Implement condition types
  - Price conditions (above, below, crosses)
  - Indicator conditions (RSI, MA, etc.)
  - Time conditions (market hours, day of week)
  - Position conditions (has position, P&L threshold)
□ Add action blocks
  - Market/limit order actions
  - Position sizing rules
  - Stop-loss/take-profit attachment
□ Create strategy validation
  - Syntax checking
  - Logic contradiction detection
  - Backtest preview
□ Implement strategy code generation
  - Export to Python strategy class
  - Readable code output
□ Add strategy templates library
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
□ Create efficient frontier visualization
  - Risk-return scatter plot
  - Frontier curve
  - Interactive point selection
□ Implement asset weight sliders
  - Manual weight adjustment
  - Constraint visualization
  - Real-time metrics update
□ Add optimization parameter panel
  - Target return/risk inputs
  - Constraint configuration (min/max weights)
  - Rebalancing frequency
□ Create correlation analysis view
  - Asset correlation matrix
  - Diversification score
  - Concentration risk metrics
□ Implement rebalancing suggestions
  - Current vs. target weights
  - Trade list generation
  - Transaction cost estimation
□ Add historical optimization analysis
  - Rolling efficient frontier
  - Regime analysis
```
**Complexity**: Medium | **Impact**: Medium

### 3.4 Paper Trading Mode

**Files**: `typescript/src-tauri/src/lib.rs`, new state management
**Status**: Simulation only, no paper trading distinction
**Impact**: Medium - Risk-free practice

```
Tasks:
□ Create paper trading account system
  - Virtual balance tracking
  - Separate from simulation
  - Persistent across sessions
□ Implement realistic execution simulation
  - Configurable fill rates
  - Slippage modeling
  - Partial fill simulation
□ Add paper trading indicator in UI
  - Clear visual distinction from live
  - "PAPER" watermark on charts
□ Create paper trading performance tracking
  - Separate P&L history
  - Comparison with live results
□ Implement paper-to-live transition
  - Strategy validation checklist
  - Risk assessment before going live
□ Add paper trading leaderboard (optional)
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
□ Create exchange abstraction layer
  - Common interface for all exchanges
  - Unified order types mapping
  - Normalized market data format
□ Implement Binance integration
  - REST API for historical data
  - WebSocket for real-time
  - Order submission (paper mode)
□ Add Kraken integration
  - Spot and futures support
  - OHLCV data fetching
□ Implement Deribit integration
  - Options data
  - Perpetual futures
□ Create exchange selector in UI
  - Connection status indicators
  - API key management per exchange
□ Add cross-exchange arbitrage view
  - Price comparison
  - Spread monitoring
```
**Complexity**: High | **Impact**: High

### 4.2 Alternative Data Sources

**Files**: new `rust/src/web/alternative/`, `python/src/data/`
**Status**: Price data only
**Impact**: Medium - Enhanced signals

```
Tasks:
□ Implement news feed integration
  - RSS aggregation
  - Keyword filtering
  - Sentiment tagging
□ Add social sentiment data
  - Twitter/X API integration
  - Reddit sentiment scraping
  - Aggregated sentiment score
□ Create on-chain data integration
  - Whale wallet tracking
  - Exchange flow monitoring
  - Network metrics (active addresses)
□ Implement economic calendar
  - Event fetching
  - Impact classification
  - Countdown timers
□ Add data source quality metrics
  - Latency monitoring
  - Data completeness scores
□ Create alternative data dashboard
```
**Complexity**: High | **Impact**: Medium

### 4.3 Historical Data Management

**Files**: new `typescript/src/components/data-manager/`, `rust/src/db/`
**Status**: Basic data fetching, no management UI
**Impact**: Medium - Data organization

```
Tasks:
□ Create data catalog UI
  - Available datasets listing
  - Metadata (date range, resolution)
  - Size and quality indicators
□ Implement data download manager
  - Batch download interface
  - Progress tracking
  - Resume capability
□ Add data quality tools
  - Gap detection
  - Outlier identification
  - Data repair suggestions
□ Create data export functionality
  - CSV export
  - Parquet export
  - Custom date range selection
□ Implement data versioning
  - Track data updates
  - Rollback capability
□ Add storage management
  - Disk usage visualization
  - Old data cleanup
```
**Complexity**: Medium | **Impact**: Medium

### 4.4 Real-Time Data Streaming Improvements

**Files**: `rust/src/web/streaming.rs`, `typescript/src-tauri/src/lib.rs`
**Status**: Basic WebSocket streaming
**Impact**: Medium - Reliability and features

```
Tasks:
□ Implement reconnection logic
  - Exponential backoff
  - State recovery after reconnect
  - Gap detection and fill
□ Add data compression
  - Delta compression for orderbook
  - Configurable compression levels
□ Create stream health monitoring
  - Latency tracking
  - Message rate monitoring
  - Drop detection
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
□ Create training job queue UI
  - Job listing with status
  - Priority management
  - Cancel/pause capability
□ Implement real-time training metrics
  - Loss curves (train/val)
  - Learning rate schedule
  - Gradient statistics
□ Add hyperparameter visualization
  - Current hyperparameters display
  - Historical comparison
□ Create GPU utilization monitor
  - Memory usage
  - Compute utilization
  - Temperature monitoring
□ Implement training logs viewer
  - Filterable log stream
  - Error highlighting
  - Export capability
□ Add early stopping controls
  - Manual stop with checkpoint
  - Patience configuration
□ Create training history browser
  - Past experiments listing
  - Metrics comparison
```
**Complexity**: Medium | **Impact**: High

### 5.2 Model Registry UI

**Files**: new `typescript/src/components/models/`
**Status**: list_trained_models command exists, no UI
**Impact**: Medium - Model management

```
Tasks:
□ Create model catalog interface
  - Model listing with metadata
  - Version history per model
  - Performance metrics summary
□ Implement model comparison view
  - Side-by-side metrics
  - Architecture differences
  - Training data differences
□ Add model deployment controls
  - Set active model
  - A/B testing configuration
  - Rollback capability
□ Create model documentation panel
  - Auto-generated model card
  - Training configuration
  - Input/output specifications
□ Implement model search and filter
  - By architecture type
  - By performance metrics
  - By creation date
□ Add model export/import
  - ONNX export
  - Model sharing
```
**Complexity**: Medium | **Impact**: Medium

### 5.3 Feature Engineering UI

**Files**: new `typescript/src/components/features/`
**Status**: Python feature pipeline exists, no UI
**Impact**: Medium - Feature development workflow

```
Tasks:
□ Create feature catalog
  - Available features listing
  - Feature statistics
  - Correlation with target
□ Implement feature builder
  - Visual feature combination
  - Mathematical operations
  - Lag/lead transformations
□ Add feature importance visualization
  - SHAP values display
  - Permutation importance
  - Feature ranking
□ Create feature validation tools
  - Distribution analysis
  - Stationarity tests
  - Look-ahead bias detection
□ Implement feature set management
  - Save/load feature sets
  - Version control
□ Add feature documentation
  - Auto-generated descriptions
  - Usage statistics
```
**Complexity**: Medium | **Impact**: Medium

### 5.4 Prediction Explanation UI

**Files**: new `typescript/src/components/explanations/`
**Status**: SHAP wrapper in Python, no UI
**Impact**: Medium - Model interpretability

```
Tasks:
□ Create prediction breakdown view
  - Feature contribution waterfall
  - Top contributing factors
  - Confidence intervals
□ Implement SHAP visualization
  - Summary plots
  - Dependence plots
  - Force plots
□ Add counterfactual explanations
  - "What-if" analysis
  - Minimum change for different outcome
□ Create attention visualization
  - For transformer models
  - Temporal attention heatmap
□ Implement prediction confidence display
  - Uncertainty quantification
  - Out-of-distribution detection
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
□ Implement auction mechanisms
  - Opening auction simulation
  - Closing auction
  - Volatility auctions
□ Add market maker simulation
  - Automated liquidity provision
  - Spread dynamics
  - Inventory management
□ Create latency simulation
  - Configurable order latency
  - Market data delay
  - Jitter modeling
□ Implement queue position tracking
  - Accurate fill simulation
  - Queue priority visualization
□ Add tick size rules
  - Price increment enforcement
  - Lot size rules
□ Create circuit breaker simulation
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
□ Create agent framework
  - Agent trait definition
  - Agent lifecycle management
  - Inter-agent messaging
□ Implement agent types
  - Momentum traders
  - Mean reversion traders
  - Market makers
  - Noise traders
□ Add agent parameterization
  - Configurable aggression
  - Capital allocation
  - Strategy parameters
□ Create market impact visualization
  - Agent activity heatmap
  - Price impact attribution
□ Implement agent performance tracking
  - Per-agent P&L
  - Market share metrics
□ Add scenario builder
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
□ Implement options order book
  - Separate books per strike/expiry
  - Options-specific order types
□ Add Greeks calculation engine
  - Real-time Greeks updates
  - Portfolio-level Greeks
□ Create options chain UI
  - Strike/expiry matrix
  - Bid/ask/last/volume
  - IV display
□ Implement exercise/assignment
  - American vs. European
  - Early exercise logic
  - Assignment simulation
□ Add volatility surface visualization
  - 3D surface plot
  - Smile/skew analysis
□ Create options strategy builder
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
□ Create scenario definition system
  - Price shock scenarios
  - Volatility spike scenarios
  - Liquidity crisis scenarios
□ Implement stress testing
  - Portfolio stress test execution
  - Multiple scenario comparison
□ Add historical scenario replay
  - Flash crash replay
  - Major event replay
□ Create Monte Carlo simulation
  - Path generation
  - VaR/CVaR calculation
  - Distribution visualization
□ Implement scenario builder UI
  - Visual scenario configuration
  - Parameter sliders
□ Add scenario results dashboard
  - P&L distribution
  - Risk metrics under stress
```
**Complexity**: High | **Impact**: Medium

---

## Implementation Roadmap

### Phase A: Core Trading Features (Weeks 1-4)
- [ ] Extended order types (1.1)
- [ ] Advanced charting features (2.1)
- [ ] Order book visualization (2.2)
- [ ] Real-time analytics dashboard (2.3)

### Phase B: Strategy Tools (Weeks 5-8)
- [ ] Visual backtesting interface (3.1)
- [ ] Training dashboard (5.1)
- [ ] Algorithmic execution engine (1.2)
- [ ] Notification system (2.4)

### Phase C: Data & Integration (Weeks 9-12)
- [ ] Multi-exchange support (4.1)
- [ ] Model registry UI (5.2)
- [ ] Historical data management (4.3)
- [ ] Paper trading mode (3.4)

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
4. **Create position P&L column** - Real-time unrealized P&L calculation
5. **Add timeframe selector** - Switch between candle intervals
6. **Implement chart crosshair** - Coordinated cursor across charts
7. **Add model list dropdown** - Simple UI for model selection
8. **Create backtest date picker** - Date range input component
9. **Add export to CSV button** - Download trade history
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

| Feature Area | Metric | Target |
|-------------|--------|--------|
| Order Types | Types supported | 10+ |
| Charts | Indicator count | 20+ |
| Backtesting | Avg backtest time | <30s for 1yr |
| Data Sources | Exchanges integrated | 4+ |
| ML UI | Training visibility | Full pipeline |
| Simulation | Agents supported | 100+ concurrent |
| UI Response | Chart render time | <16ms (60fps) |
| Strategy Builder | Rule types | 15+ |

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

*Last Updated: 2026-01-21*
*Version: 3.0 (Feature-Focused Plan)*
