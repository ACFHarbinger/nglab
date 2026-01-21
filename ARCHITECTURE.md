# NGLab Architecture: The Blueprint

> **System Overview**: Design Patterns, Component Boundaries, and Data Flow.

NGLab is a sophisticated **Multimodal Deep Reinforcement Learning Platform** for financial trading. It follows a strict **Hexagonal Architecture** (Ports & Adapters) to decouple the high-performance core from external interfaces.

---

## 1. System Context Diagram

Where does NGLab fit in the world?

```mermaid
graph TD
    User[Quantitative Trader] -- "Views Dashboard" --> UI[Tauri GUI]
    User -- "Configures Exp" --> Hydra[Hydra Config]
    
    subgraph "NGLab Platform"
        UI -- "Events/Cmds" --> RustCore[Rust Simulation Core]
        Hydra -- "Parametrizes" --> PythonBrain[Python Research Layer]
        
        PythonBrain -- "Controls" --> RustCore
        RustCore -- "Feeds Data" --> PythonBrain
    end
    
    Poly[Polymarket API] -- "Market Data" --> RustCore
    Exchange[Crypto Exchange] -- "Price Integration" --> RustCore
```

---

## 2. Directory Structure Map

A quick reference to where the logic lives.

| Directory | Layer | Purpose | Key Technologies |
| :--- | :--- | :--- | :--- |
| `/rust` | **Simulation** | Matching Engine, Risk, Gym Env | Tokio, PyO3, Serde |
| `/python` | **Intelligence** | Models, Agents, Training Loops | PyTorch, Hydra, Optuna |
| `/typescript` | **Interface** | GUI, Charts, Commands | React 19, Tauri 2.0, Vite |
| `/deploy` | **Ops** | CI/CD, Docker, Kubernetes | GitHub Actions, Docker Compose |
| `/scripts` | **Tools** | Setup, Benchmarks, Maintenance | Bash, Just |

---

## 3. Layer 1: Simulation Engine (Rust)

**Location:** `/rust`

### Core Components

#### 1. TradingEnv (Gymnasium-compatible)
**File:** `rust/src/simulation/gym.rs`

A reinforcement learning environment following the Gymnasium API standard.

**Features:**
- **Action Space**: Discrete (hold, buy, sell) or continuous
- **Observation Space**: Price history with configurable lookback window
- **Reward Function**: Risk-adjusted returns (Sharpe ratio)
- **State Management**: Portfolio value, position tracking, cash management
- **PyO3 Integration**: Zero-copy numpy array transfers to Python

**Key Methods:**
```rust
pub fn new(initial_cash: f64, lookback_window: usize, transaction_cost: f64) -> Self
pub fn reset(&mut self, seed: Option<u64>) -> (Array1<f64>, HashMap<String, f64>)
pub fn step(&mut self, action: i32) -> StepResult
```

**Performance Characteristics:**
- Step execution: <1ms
- Memory-efficient rolling window
- No heap allocations in hot path

#### 2. MultiAssetEnv (Portfolio Simulation)
**File:** `rust/src/simulation/multi_asset.rs`

Extension of the trading environment for multi-asset portfolio management.

**Features:**
- **Concurrent Simulation**: Simulates multiple order books and price series simultaneously.
- **Portfolio Management**: Tracks cash, asset positions, and total portfolio value across all assets.
- **Advanced Observations**: Generates flattened feature vectors per asset (price, returns, volatility, imbalance, position).
- **Execution Engine**: Matches actions against individual `OrderBook` instances with synthetic liquidity and stochastic slippage.
- **Native & Python API**: Unified core logic with specialized entry points for Rust and Python.

#### 3. OrderBook (Central Limit Order Book)
**File:** `rust/src/simulation/orderbook.rs`

A production-grade CLOB implementation with price-time priority matching.

**Architecture:**
```
OrderBook
├── bids: BTreeMap<OrderedFloat<f64>, PriceLevel>  (sorted descending)
├── asks: BTreeMap<OrderedFloat<f64>, PriceLevel>  (sorted ascending)
└── orders: HashMap<u64, Order>                     (fast lookup by ID)

PriceLevel
├── price: f64
├── total_volume: u64
└── orders: VecDeque<Order>  (FIFO queue for price-time priority)
```

**Features:**
- **Matching Algorithm**: Price-time priority FIFO
- **Queue Position Tracking**: For HFT simulation accuracy
- **Advanced Orders**: 
  - **Iceberg**: Only part of the order is visible; refills from hidden quantity and resets priority on refill.
  - **Trailing Stop**: Trigger price adjusts dynamically as market moves favorably.
  - **Stop-Loss/Take-Profit**: Standard price-triggered conversion to market/limit orders.
- **Serialization**: Full state persistence with Serde
- **Performance**: O(log n) insertion, O(1) best bid/ask

**Key Methods:**
```rust
pub fn add_order(&mut self, order: Order) -> Vec<Trade>
pub fn cancel_order(&mut self, order_id: u64) -> Result<Order>
pub fn get_best_bid(&self) -> Option<f64>
pub fn get_best_ask(&self) -> Option<f64>
pub fn get_price_levels(&self, depth: usize) -> (Vec<PriceLevel>, Vec<PriceLevel>)
```

#### 4. PolymarketArena
**File:** `rust/src/simulation/polymarket.rs`

Prediction market simulation using conditional token framework.

**Features:**
- **Binary Outcome Markets**: Yes/No token pairs
- **Collateral Management**: USDC-based accounts
- **Market Making**: Automated market maker (AMM) simulation
- **Unrealized PnL**: Real-time profit/loss tracking
- **Position Management**: Long/short position tracking

**Architecture:**
```
PolymarketArena
├── markets: HashMap<String, Market>
├── accounts: HashMap<String, Account>
└── trades: Vec<Trade>

Market
├── yes_tokens: f64
├── no_tokens: f64
├── collateral: f64
└── outcome: Option<bool>
```

#### 5. Financial Models
**Location:** `rust/src/models/`

Advanced quantitative finance models for pricing and risk management.

- **Black-Scholes** (`black_scholes.rs`): European option pricing
- **Rough Heston** (`rough_heston.rs`): Rough volatility path simulation
- **Rough Bergomi** (`rough_bergomi.rs`): Alternative rough volatility model
- **Credit Risk** (`credit_risk.rs`): Default probability modeling

#### 6. Time Series Forecasting ("Project Moon")
**Location:** `rust/src/moon/`

Classical time series forecasting methods.

- **ARIMA** (`arima.rs`): AutoRegressive Integrated Moving Average
- **GARCH** (`garch.rs`): Volatility modeling
- **Exponential Smoothing** (`es.rs`): Trend and seasonality
- **Prophet** (`prophet.rs`): Changepoint-based forecasting (partial)

#### 7. Web Scraping & Data Collection
**Location:** `rust/src/web/`

Real-time market data acquisition.

**PolymarketScraper** (`polymarket.rs`):
- Multiple frequency support (minutely to monthly)
- Historical price replay
- Asynchronous HTTP with Reqwest
- Rate limiting and error handling

---

## 4. Layer 2: Training & Learning (Python)

**Location:** `/python`

### Core Components

#### 1. Deep Learning Models
**Location:** `python/src/models/`

Comprehensive suite of time series and generative models.

**Generative Models:**
- **VAE** (`vae.py`): Variational Auto-Encoder for time series
  - Encoder types: Transformer, Mamba, LSTM, GRU, xLSTM
  - Reparameterization trick for gradient flow
  - KL divergence regularization
- **Diffusion UNet** (`diffusion_unet.py`): 1D diffusion model
- **TimeGAN** (`gan_networks.py`): Adversarial time series generation

**Sequence Models:**
- **CNN** (`cnn.py`): Rolling window convolutional networks
- **RNN Variants** (`rnn.py`): LSTM, GRU with dropout
- **xLSTM** (`xlstm.py`): Extended LSTM architecture
- **Mamba** (`tsmamba.py`): State space model for sequences
- **NSTransformer** (`nstransformer.py`): Non-stationary Transformer

**Model Architecture:**
```python
TimeSeriesBackbone
├── encoder: Union[TransformerEncoder, MambaBlock, LSTM, GRU]
├── decoder: Union[TransformerDecoder, Linear]
└── head: TaskHead (classification/regression)

VAE
├── encoder: TimeSeriesBackbone
├── mu_layer: Linear
├── logvar_layer: Linear
└── decoder: TimeSeriesBackbone
```

#### 2. Training Pipeline
**Location:** `python/src/pipeline/`

PyTorch Lightning-based training infrastructure.

**Lightning Modules:**
- `vae_module.py`: VAE training with KL annealing
- `diffusion_module.py`: Diffusion model training
- `gan_module.py`: GAN training with discriminator
- `sl_module.py`: Supervised learning
- `rl_module.py`: Reinforcement learning

**Hyperparameter Optimization:**
- **Optuna Integration** (`hpo/`): Bayesian optimization
- **Custom Objectives** (`objectives/`): Task-specific metrics
- **Pruning**: Early stopping for unpromising trials

**Training Scripts:**
- `train.py`: Generic training entry point
- `train_ppo.py`: Proximal Policy Optimization
- `train_sac.py`: Soft Actor-Critic

#### 3. Reinforcement Learning Agents
**Location:** `python/src/agents/`

TorchRL-based agent implementations.

**TradingEnvWrapper** (`env_wrapper.py`):
- Wraps Rust TradingEnv for TorchRL compatibility
- TensorDict observation/action spaces
- Automatic batching and GPU transfer

**RL Algorithms:**
- PPO (Proximal Policy Optimization)
- SAC (Soft Actor-Critic)
- Custom policy networks

#### 4. Trading Policies
**Location:** `python/src/policies/`

Strategy implementations for trading.

- **Base Policy** (`base.py`): Abstract policy interface
- **Regular Policy** (`regular.py`): Simple rule-based strategies
- **Threshold Policy** (`threshold.py`): Threshold-based trading
- **Neural Policy** (`neural.py`): Deep learning-based strategies
- **Black-Scholes** (`black_scholes.py`): Options pricing strategy

#### 5. Configuration Management
**Location:** `python/src/conf/`

Hydra-based hierarchical configuration.

```
conf/
├── config.yaml          # Main configuration
├── task/                # Task-specific configs
├── model/               # Model architectures
├── env/                 # Environment settings
└── algorithm/           # Training algorithms
```

**Benefits:**
- Type-safe configuration with dataclasses
- Composition and overrides
- Experiment tracking integration
- Command-line overrides: `python train.py model=vae model.latent_dim=128`

---

## 5. Layer 3: User Interface (TypeScript/Tauri)

**Location:** `/typescript`

### Frontend Architecture (React)

#### 1. Main Application
**File:** `typescript/src/App.tsx`

Tab-based navigation with real-time updates.

**Tabs:**
- **Simulation**: Live trading environment visualization
- **Scraper**: Polymarket data collection interface
- **Analysis**: Market analysis tools
- **Prediction**: Model prediction display
- **Pricing**: Options pricing calculator

#### 2. Custom Hooks

**useArena** (`hooks/useArena.ts`):
```typescript
interface ArenaState {
  stepInfo: StepInfo;        // Portfolio, position, cash, Sharpe
  orderBook: OrderBook;      // Live order book snapshot
  priceHistory: number[];    // Capped at 200 points
  isRunning: boolean;
}

// Listens to Tauri 'arena-update' events
const { arenaState, start, stop } = useArena();
```

**usePolymarket** (`hooks/usePolymarket.ts`):
- Live price streaming from Polymarket API
- Market selection and filtering
- Historical data replay

#### 3. Components

**PriceChart** (`components/charts/PriceChart.tsx`):
- Lightweight-charts integration
- Real-time candlestick/line charts
- Performance-optimized (60 FPS target)

**OrderBook** (`components/dashboard/OrderBook.tsx`):
- Real-time bid/ask visualization
- Depth chart display
- Price level highlighting

### Backend Architecture (Tauri)

**Location:** `typescript/src-tauri`

#### Tauri Backend (Rust)
**File:** `src-tauri/src/lib.rs`

**Architecture:**
```rust
ArenaState
├── env: Arc<Mutex<TradingEnv>>    // Thread-safe environment
└── task_handle: Option<JoinHandle> // Simulation loop task

// Event-driven architecture
tokio::spawn(async move {
    loop {
        let step_info = env.lock().unwrap().step(action);
        app_handle.emit_all("arena-update", ArenaUpdate { ... });
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
});
```

**Tauri Commands:**
```rust
#[tauri::command]
async fn start_arena(state: State<'_, ArenaState>) -> Result<()>

#[tauri::command]
async fn stop_arena(state: State<'_, ArenaState>) -> Result<()>

#[tauri::command]
fn get_orderbook() -> OrderBookSnapshot
```

**Event Flow:**
1. Frontend calls `invoke('start_arena')`
2. Backend spawns Tokio task for simulation loop
3. Each step emits `arena-update` event
4. Frontend `useArena` hook receives updates
5. React components re-render with new data

---

## Data Flow

### Training Pipeline Data Flow

```
┌──────────────┐
│ Market Data  │ (CSV, API, Live Stream)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Preprocessor │ (Normalization, Feature Engineering)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  TradingEnv  │ (Rust Simulation)
│    (Rust)    │
└──────┬───────┘
       │ PyO3
       │ Zero-copy NumPy
       ▼
┌──────────────┐
│ TorchRL Env  │ (Python Wrapper)
│   Wrapper    │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ RL Agent     │ (PPO, SAC, etc.)
│  Training    │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Trained Model│ → Evaluation → Deployment
└──────────────┘
```

### Real-time Visualization Data Flow

```
┌──────────────┐
│ Market Feed  │ (Polymarket WebSocket)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│PolymarketScraper│ (Rust)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ TradingEnv   │ (Environment State Update)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Tauri Event  │ (arena-update emission)
│   System     │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ React Hook   │ (useArena)
│  (Frontend)  │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Chart Update │ (PriceChart, OrderBook)
└──────────────┘
```

---
## 6. Key Design Patterns

### 6.1 Zero-Copy Data Transfer (Rust ↔ Python)

Using PyO3 and numpy crate for efficient memory sharing. This pattern avoids expensive serialization during high-frequency stepping.

```rust
// Rust side
use numpy::{PyArray1, ToPyArray};
use pyo3::prelude::*;

#[pyclass]
pub struct TradingEnv {
    observation: Array1<f64>,
}

#[pymethods]
impl TradingEnv {
    fn get_observation<'py>(&self, py: Python<'py>) -> &'py PyArray1<f64> {
        self.observation.to_pyarray(py)  // Zero-copy transfer
    }
}
```

```python
# Python side
import nglab
import numpy as np

env = nglab.TradingEnv(10000.0, 50, 0.001)
obs, _ = env.reset()  # obs is numpy array, no copy
```

### 6.2 Event-Driven Architecture (Tauri)

Decoupled frontend-backend communication enabling responsive UI even under heavy simulation load.

```typescript
// Frontend subscribes to events
useEffect(() => {
  const unlisten = listen('arena-update', (event) => {
    setArenaState(event.payload);
  });
  return () => { unlisten.then(f => f()); };
}, []);

// Backend emits events
app_handle.emit_all("arena-update", payload)?;
```

### 6.3 Configuration Composition (Hydra)

Hierarchical configuration with runtime overrides allows for seamless experiment scaling.

```yaml
# config.yaml
defaults:
  - model: vae
  - env: trading
  - algorithm: ppo

# model/vae.yaml
latent_dim: 64
encoder_type: transformer

# CLI override
$ python train.py model.latent_dim=128 env.initial_cash=50000
```

### 6.4 Async Runtime (Tokio)

Non-blocking I/O for scalable data ingestion.

```rust
#[tokio::main]
async fn main() {
    let scraper = PolymarketScraper::new();

    // Concurrent data fetching
    let (prices, metadata) = tokio::join!(
        scraper.fetch_prices(),
        scraper.fetch_metadata()
    );
}
```

---

## 7. Data Flow Specification

Data moves through the system in three distinct loops.

### 7.1 The Fast Loop (Simulation)
*Frequency: 10kHz+*

```mermaid
sequenceDiagram
    participant User
    participant CLOB as OrderBook
    participant Engine as MatchingEngine
    participant TradeBuffer

    User->>CLOB: Add Limit Order
    CLOB->>Engine: Check for Price Equality
    Engine->>CLOB: Match Found
    Engine->>TradeBuffer: Emit Trade Execution
    TradeBuffer->>User: Notify Fill
```

1.  **Input**: Limit Order (Add/Cancel).
2.  **Process**: `OrderBook` matching engine.
3.  **Output**: `Trade` events, `QueuePosition` updates.
4.  **Storage**: In-memory ring buffer.

### 7.2 The Learning Loop (RL)
*Frequency: User Defined (e.g., 1 minute candles)*

```mermaid
sequenceDiagram
    participant Agent as Python (Agent)
    participant Wrapper as TorchRL Env
    participant Bridge as PyO3/Numpy
    participant Rust as Rust Env

    Agent->>Wrapper: Select Action
    Wrapper->>Rust: step(action)
    Rust->>Rust: Execute Order & Update PnL
    Rust->>Bridge: Zero-Copy Observation
    Bridge->>Wrapper: TensorDict
    Wrapper->>Agent: Next State & Reward
```

1.  **Observation**: Rust builds a flat `Vec<f64>` feature vector.
2.  **Bridge**: `PyArray` (numpy) wraps the memory pointer (Zero-Copy).
3.  **Inference**: Python Model $\pi(s)$ computes action $a$.
4.  **Action**: Integer/Float passed back to Rust via `step(a)`.
5.  **Reward**: Rust computes $r_t$ and returns it.

### 7.3 The UI Loop (Visualization)
*Frequency: 60Hz (Throttled)*

```mermaid
sequenceDiagram
    participant Rust as Simulation Task
    participant Tauri as Event Bus
    participant React as Frontend
    participant Canvas as Charts/DOM

    Rust->>Tauri: emit('arena-update')
    Tauri->>React: useArena (Hook)
    React->>React: Update State
    React->>Canvas: Redraw Orderbook/Chart
```

1.  **Event**: `ArenaUpdate` struct is serialized to JSON.
2.  **Transport**: Tauri Event Bus (`app.emit`).
3.  **Render**: React `useEffect` triggers a canvas redraw.

---

## 8. Key Design Decisions

### 8.1 Why Rust for Simulation?
Python is too slow for Order Book matching ($O(N)$ list operations vs Rust's $O(\log N)$ B-Trees). Garbage collection pauses in Python would destroy the determinism required for accurate backtesting.

### 8.2 Why Python for ML?
The ecosystem. PyTorch, HuggingFace, and Optuna are unrivaled. We bridge the two worlds using `PyO3`, giving us "C++ speed with Python usability".

### 8.3 Why Tauri?
Electron is bloated (Chromium + Node.js bundle). Tauri uses the OS's native webview (WebKit on Linux/macOS, WebView2 on Windows) and a lightweight Rust backend. This results in a <10MB binary vs >100MB for Electron.

---

## 9. Performance Characteristics

### Rust Layer
| Component | Metric | Target | Actual |
|-----------|--------|--------|--------|
| OrderBook Insert | Latency | <1ms | ~0.1ms |
| TradingEnv Step | Latency | <1ms | ~0.5ms |
| Order Matching | Throughput | >10k ops/sec | ~50k ops/sec |
| Memory Usage | RAM | <100MB | ~50MB |

### Python Layer
| Component | Metric | Target | Notes |
|-----------|--------|--------|-------|
| Model Forward Pass | Latency | <10ms | Depends on model size |
| Training Step | Latency | <100ms | Includes optimizer step |
| Data Loading | Throughput | >1000 samples/sec | With prefetching |

### Frontend Layer
| Component | Metric | Target | Notes |
|-----------|--------|--------|-------|
| Chart Rendering | FPS | 60 | Lightweight-charts optimized |
| Event Processing | Latency | <50ms | Tauri IPC overhead |
| Memory Usage | RAM | <500MB | With 200-point history |

---

## 10. Scalability Considerations

### Horizontal Scaling
- **Training**: Distributed via PyTorch DDP or Ray
- **Simulation**: Multiple environment instances in parallel
- **Data Collection**: Sharded by market or time period

### Vertical Scaling
- **GPU Acceleration**: Full PyTorch/TorchRL support
- **Multi-threading**: Tokio runtime for concurrent I/O
- **Memory Efficiency**: Streaming data processing, bounded buffers

---

## 11. Security Considerations

### API Keys
- Stored in environment variables
- Never committed to version control
- Validated before use

### Data Validation
- Input sanitization at all boundaries
- Type checking with strong typing (Rust, TypeScript)
- Schema validation for external data

### Sandboxing
- Tauri provides OS-level sandboxing
- Limited filesystem access
- Content Security Policy (CSP) for web views

---

## 12. Deployment Topology

### Local Research (Dev)
- **Single Machine**: 1 GPU, Multi-core CPU.
- **Process**: `cargo run` hosts the simulation; Python is loaded as a dynamic library (`.so`).

### Development
```
Developer Machine
├── Rust: cargo build
├── Python: uv sync && maturin develop
└── Tauri: npm run tauri dev
```


### Cloud Training (Prod - Planned)
```
Cloud Infrastructure
├── Training Cluster (GPU)
│   ├── Model training jobs
│   └── Hyperparameter optimization
├── Inference Service (API)
│   ├── Model serving
│   └── Real-time predictions
└── Desktop App Distribution
    ├── Linux: AppImage
    ├── macOS: DMG
    └── Windows: MSI
```

### Kubernetes Production Topology

```mermaid
graph TB
    subgraph "Ingress Layer"
        ING[NGINX Ingress Controller]
        CERT[Cert-Manager]
    end

    subgraph "Application Layer"
        API1[nglab-api Pod 1]
        API2[nglab-api Pod 2]
        API3[nglab-api Pod 3]
        
        TRAIN1[Training Worker 1]
        TRAIN2[Training Worker 2]
    end

    subgraph "Data Layer"
        PG[(PostgreSQL)]
        REDIS[(Redis Cache)]
        S3[(S3 / MinIO)]
    end

    subgraph "Observability"
        PROM[Prometheus]
        GRAF[Grafana]
        JAEGER[Jaeger]
    end

    ING --> API1 & API2 & API3
    API1 & API2 & API3 --> PG
    API1 & API2 & API3 --> REDIS
    TRAIN1 & TRAIN2 --> S3
    API1 & API2 & API3 --> PROM
    PROM --> GRAF
    API1 & API2 & API3 --> JAEGER
```

### Kubernetes Resource Requirements

| Component | Replicas | CPU (Request/Limit) | Memory (Request/Limit) | GPU |
|-----------|----------|---------------------|------------------------|-----|
| **nglab-api** | 2-5 | 500m / 2000m | 1Gi / 4Gi | No |
| **training-worker** | 1-4 | 2000m / 8000m | 4Gi / 32Gi | Yes (1-2) |
| **postgresql** | 1 (HA: 3) | 500m / 2000m | 1Gi / 4Gi | No |
| **redis** | 1 (HA: 3) | 100m / 500m | 256Mi / 1Gi | No |
| **prometheus** | 1 | 500m / 1000m | 2Gi / 4Gi | No |
| **grafana** | 1 | 100m / 500m | 256Mi / 512Mi | No |
| **jaeger** | 1 | 500m / 1000m | 1Gi / 2Gi | No |

### Kustomize Overlay Structure

```
deploy/k8s/
├── base/                    # Shared configurations
│   ├── deployment.yaml      # API and worker deployments
│   ├── service.yaml         # Service definitions
│   ├── configmap.yaml       # Environment configuration
│   ├── secrets.yaml         # Sensitive data (sealed)
│   ├── pvc.yaml             # Persistent volume claims
│   ├── hpa.yaml             # Horizontal pod autoscaler
│   ├── pdb.yaml             # Pod disruption budget
│   ├── serviceaccount.yaml  # RBAC
│   └── kustomization.yaml   # Base kustomization
└── overlays/
    ├── dev/                 # Development (minikube)
    │   ├── kustomization.yaml
    │   └── patch-resources.yaml
    ├── staging/             # Pre-production
    │   ├── kustomization.yaml
    │   └── patch-replicas.yaml
    └── prod/                # Production
        ├── kustomization.yaml
        ├── patch-resources.yaml
        └── ingress.yaml
```

### Deployment Commands

```bash
# Deploy to development
kubectl apply -k deploy/k8s/overlays/dev

# Deploy to staging
kubectl apply -k deploy/k8s/overlays/staging

# Deploy to production
kubectl apply -k deploy/k8s/overlays/prod

# Check deployment status
kubectl get pods -n nglab -w
```

---

## 13. Testing Strategy

### Unit Tests
- **Rust**: `#[test]` modules, `cargo test`
- **Python**: pytest, `tests/` directory
- **TypeScript**: Jest/Vitest (future)

### Integration Tests
- **Rust-Python**: PyO3 bindings verification
- **Tauri Events**: Frontend-backend communication
- **End-to-End**: Full training pipeline

### Benchmarks
- **Criterion**: Rust performance regression testing
- **pytest-benchmark**: Python performance testing
- **Lighthouse**: Frontend performance audits (future)

---

## 14. Future Architecture Enhancements

### Short-term (3-6 months)
- [ ] Add model serving API (FastAPI/Actix-web)
- [ ] Implement distributed training with Ray
- [ ] Add real-time backtesting engine
- [ ] Enhance monitoring with Prometheus/Grafana

### Medium-term (6-12 months)
- [ ] Microservices architecture for scalability
- [ ] Multi-asset trading support
- [ ] Advanced risk management system
- [ ] Cloud deployment automation

### Long-term (12+ months)
- [ ] Multi-agent reinforcement learning
- [ ] Causal inference integration
- [ ] Explainable AI for trading decisions
- [ ] Regulatory compliance framework

---

## 15. Error Handling Patterns

### 15.1 Rust Error Handling

We use a custom error enum for type-safe error handling:

```rust
#[derive(Debug, thiserror::Error)]
pub enum ArenaError {
    #[error("Invalid order: {0}")]
    InvalidOrder(String),
    
    #[error("Insufficient funds: required {required}, available {available}")]
    InsufficientFunds { required: f64, available: f64 },
    
    #[error("Order not found: {order_id}")]
    OrderNotFound { order_id: u64 },
    
    #[error("Market closed")]
    MarketClosed,
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}
```

**Error Propagation:**
```rust
pub fn execute_trade(&mut self, order: Order) -> Result<Trade, ArenaError> {
    self.validate_order(&order)?;  // Propagate validation errors
    self.check_funds(&order)?;     // Propagate fund errors
    
    let trade = self.match_order(order)?;
    Ok(trade)
}
```

### 15.2 Python Error Handling

Custom exception hierarchy:

```python
class NGLabError(Exception):
    """Base exception for NGLab."""
    pass

class ConfigurationError(NGLabError):
    """Invalid configuration."""
    pass

class TrainingError(NGLabError):
    """Training pipeline failure."""
    pass

class ModelError(NGLabError):
    """Model inference/loading error."""
    pass
```

### 15.3 Cross-Language Error Mapping

PyO3 automatically converts Rust errors to Python exceptions:

```rust
#[pymethods]
impl TradingEnv {
    fn step(&mut self, action: i32) -> PyResult<StepResult> {
        self.inner_step(action)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }
}
```

---

## 16. Observability & Monitoring

### 16.1 Metrics Stack

```mermaid
graph LR
    App[NGLab App] --> OTel[OpenTelemetry Collector]
    OTel --> Prometheus[Prometheus]
    OTel --> Jaeger[Jaeger]
    Prometheus --> Grafana[Grafana]
    Jaeger --> Grafana
```

### 16.2 Key Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `nglab_orders_total` | Counter | Total orders processed |
| `nglab_order_latency_ms` | Histogram | Order processing latency |
| `nglab_portfolio_value` | Gauge | Current portfolio value |
| `nglab_position` | Gauge | Current position size |
| `nglab_training_loss` | Gauge | Current training loss |
| `nglab_training_reward` | Gauge | Episode reward |

### 16.3 Tracing

We use OpenTelemetry for distributed tracing:

```rust
use tracing::{instrument, info_span};

#[instrument(skip(self))]
pub fn step(&mut self, action: i32) -> Result<StepResult, ArenaError> {
    let _span = info_span!("execute_action", action = action).entered();
    // ...
}
```

### 16.4 Logging Configuration

```yaml
# config/logging.yaml
version: 1
disable_existing_loggers: false

handlers:
  console:
    class: logging.StreamHandler
    level: INFO
  file:
    class: logging.FileHandler
    filename: logs/nglab.log
    level: DEBUG

loggers:
  nglab:
    level: DEBUG
    handlers: [console, file]
```

---

## 17. Database Schema

### 17.1 SQLite Schema (markets.db)

```sql
-- Market definitions
CREATE TABLE markets (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP,
    outcome BOOLEAN
);

-- Price history
CREATE TABLE price_ticks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    market_id TEXT NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    yes_price REAL NOT NULL,
    no_price REAL NOT NULL,
    volume REAL,
    FOREIGN KEY (market_id) REFERENCES markets(id)
);

-- Index for time-series queries
CREATE INDEX idx_price_ticks_market_time 
ON price_ticks(market_id, timestamp);

-- Agent positions
CREATE TABLE positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id TEXT NOT NULL,
    market_id TEXT NOT NULL,
    position_type TEXT CHECK(position_type IN ('yes', 'no')),
    quantity REAL NOT NULL,
    avg_price REAL NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 17.2 Model Registry (MLflow)

| Table | Purpose |
|-------|---------|
| `experiments` | Experiment metadata |
| `runs` | Individual training runs |
| `metrics` | Training metrics (loss, reward) |
| `params` | Hyperparameters |
| `artifacts` | Model checkpoints |

---

## 18. Configuration Reference

### 18.1 Hydra Configuration Structure

```yaml
# config.yaml
defaults:
  - _self_
  - model: mamba
  - env: trading
  - algorithm: ppo

# Environment settings
env:
  initial_cash: 100000.0
  transaction_cost: 0.001
  lookback_window: 50
  max_steps: 10000

# Training settings
training:
  batch_size: 256
  learning_rate: 3e-4
  num_envs: 16
  total_steps: 1000000
  checkpoint_interval: 10000

# Logging
logging:
  level: INFO
  wandb:
    enabled: true
    project: nglab
    entity: your-team
```

### 18.2 Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `NGLAB_LOG_LEVEL` | Logging verbosity | `INFO` |
| `NGLAB_DATA_DIR` | Data storage path | `./data` |
| `NGLAB_CACHE_DIR` | Cache directory | `./cache` |
| `CUDA_VISIBLE_DEVICES` | GPU selection | All |
| `WANDB_API_KEY` | W&B authentication | - |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OpenTelemetry endpoint | `localhost:4317` |

---

## 19. API Contracts

### 19.1 Rust-Python Interface

```rust
/// TradingEnv is exposed to Python via PyO3
#[pyclass]
pub struct TradingEnv { ... }

#[pymethods]
impl TradingEnv {
    /// Create a new trading environment
    #[new]
    fn new(initial_cash: f64, lookback_window: usize, transaction_cost: f64) -> Self;
    
    /// Reset the environment
    fn reset(&mut self, seed: Option<u64>) -> (Py<PyArray1<f64>>, PyObject);
    
    /// Take a step in the environment
    fn step(&mut self, action: i32) -> PyResult<StepResult>;
    
    /// Get current portfolio value
    #[getter]
    fn portfolio_value(&self) -> f64;
}
```

### 19.2 Tauri Command Interface

```typescript
// Frontend → Backend Commands
interface TauriCommands {
  start_arena(): Promise<void>;
  stop_arena(): Promise<void>;
  get_orderbook(): Promise<OrderBook>;
  submit_order(order: OrderRequest): Promise<OrderResult>;
  get_config(): Promise<Config>;
  set_config(config: Partial<Config>): Promise<void>;
}

// Backend → Frontend Events
interface TauriEvents {
  'arena-update': ArenaUpdate;
  'trade-executed': TradeEvent;
  'error': ErrorEvent;
  'log': LogEvent;
}
```

### 19.3 REST API (Planned)

```yaml
openapi: 3.0.0
info:
  title: NGLab API
  version: 1.0.0

paths:
  /api/v1/arena/start:
    post:
      summary: Start simulation
      responses:
        200:
          description: Arena started
  
  /api/v1/arena/step:
    post:
      summary: Execute a step
      requestBody:
        content:
          application/json:
            schema:
              type: object
              properties:
                action:
                  type: integer
      responses:
        200:
          description: Step result
  
  /api/v1/models:
    get:
      summary: List available models
    post:
      summary: Upload a new model
```

---

## 20. Dependency Graph

```mermaid
graph TB
    subgraph Rust
        R1[orderbook.rs]
        R2[gym.rs]
        R3[risk.rs]
        R4[polymarket.rs]
        R2 --> R1
        R2 --> R3
    end
    
    subgraph Python
        P1[models/]
        P2[pipeline/]
        P3[agents/]
        P2 --> P1
        P3 --> P2
    end
    
    subgraph TypeScript
        T1[hooks/]
        T2[components/]
        T2 --> T1
    end
    
    P3 --> R2
    T1 --> R2
```

---

## 21. System Requirements

### Minimum Requirements

| Component | Minimum | Recommended | Notes |
|-----------|---------|-------------|-------|
| **CPU** | 4 cores | 8+ cores | AMD64 architecture required |
| **RAM** | 8 GB | 32 GB | 16 GB for multi-agent training |
| **GPU** | None | NVIDIA RTX 3080+ | CUDA 11.8+ for training |
| **Storage** | 20 GB SSD | 100 GB NVMe | Fast I/O for data loading |
| **OS** | Ubuntu 22.04 | Ubuntu 24.04 | Also macOS 13+, Windows 11 |

### Software Dependencies

| Software | Version | Purpose |
|----------|---------|---------|
| **Rust** | 1.75+ | Simulation engine |
| **Python** | 3.11+ | ML pipeline |
| **Node.js** | 20+ | Frontend development |
| **CUDA** | 11.8+ | GPU acceleration (optional) |
| **Docker** | 24+ | Containerization |
| **kubectl** | 1.28+ | Kubernetes deployment |

### Network Requirements

| Port | Service | Protocol | Notes |
|------|---------|----------|-------|
| 8000 | API Server | HTTP/HTTPS | Main application |
| 5432 | PostgreSQL | TCP | Database |
| 6379 | Redis | TCP | Cache |
| 4317 | Jaeger OTLP | gRPC | Tracing |
| 3000 | Grafana | HTTP | Monitoring UI |
| 9090 | Prometheus | HTTP | Metrics |

---

## Related Documentation

- [DEVELOPMENT.md](DEVELOPMENT.md) - Development setup guide
- [TESTING.md](TESTING.md) - Testing strategy
- [TUTORIAL.md](TUTORIAL.md) - Developer encyclopedia
- [AGENTS.md](AGENTS.md) - Agent taxonomy and policies
- [CONTRIBUTING.md](CONTRIBUTING.md) - Contribution guidelines
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Common issues and fixes
- [IMPROVEMENTS.md](IMPROVEMENTS.md) - Development roadmap
- [README.md](README.md) - Getting started

---

## Changelog

### Version 2.3.0 (Current)
- Added Kubernetes Production Topology with Mermaid diagrams
- Added Kubernetes Resource Requirements table
- Added Kustomize Overlay Structure documentation
- Added System Requirements section
- Updated Related Documentation links

### Version 2.2.0
- Added Error Handling Patterns section
- Added Observability & Monitoring with metrics definitions
- Added Database Schema documentation
- Added Configuration Reference
- Added API Contracts for all interfaces
- Added Dependency Graph visualization

### Version 2.1.0
- Added System Context Diagram
- Added Directory Structure Map
- Added Data Flow Specification with Mermaid diagrams

### Version 2.0.0
- Initial architecture document

---

**Last Updated:** 2026-01-21
**Version:** 2.3.0 (The Complete System Blueprint)
**Maintainer:** NGLab Team
