# NGLab Architecture: The Blueprint

<a href="https://www.gnu.org/licenses/agpl-3.0"><img alt="License: AGPL v3" src="https://img.shields.io/badge/License-AGPL_v3-blue.svg"></a>

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

| Directory     | Layer            | Purpose                        | Key Technologies               |
| :------------ | :--------------- | :----------------------------- | :----------------------------- |
| `/rust`       | **Simulation**   | Matching Engine, Risk, Gym Env | Tokio, PyO3, Serde             |
| `/python`     | **Intelligence** | Models, Agents, Training Loops | PyTorch, Hydra, Optuna         |
| `/typescript` | **Interface**    | GUI, Charts, Commands          | React 19, Tauri 2.0, Vite      |
| `/infrastructure`     | **Ops**          | CI/CD, Docker, Kubernetes      | GitHub Actions, Docker Compose |
| `/scripts`    | **Tools**        | Setup, Benchmarks, Maintenance | Bash, Just                     |

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
  stepInfo: StepInfo; // Portfolio, position, cash, Sharpe
  orderBook: OrderBook; // Live order book snapshot
  priceHistory: number[]; // Capped at 200 points
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

_Frequency: 10kHz+_

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

_Frequency: User Defined (e.g., 1 minute candles)_

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

_Frequency: 60Hz (Throttled)_

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

| Component        | Metric     | Target       | Actual       |
| ---------------- | ---------- | ------------ | ------------ |
| OrderBook Insert | Latency    | <1ms         | ~0.1ms       |
| TradingEnv Step  | Latency    | <1ms         | ~0.5ms       |
| Order Matching   | Throughput | >10k ops/sec | ~50k ops/sec |
| Memory Usage     | RAM        | <100MB       | ~50MB        |

### Python Layer

| Component          | Metric     | Target            | Notes                   |
| ------------------ | ---------- | ----------------- | ----------------------- |
| Model Forward Pass | Latency    | <10ms             | Depends on model size   |
| Training Step      | Latency    | <100ms            | Includes optimizer step |
| Data Loading       | Throughput | >1000 samples/sec | With prefetching        |

### Frontend Layer

| Component        | Metric  | Target | Notes                        |
| ---------------- | ------- | ------ | ---------------------------- |
| Chart Rendering  | FPS     | 60     | Lightweight-charts optimized |
| Event Processing | Latency | <50ms  | Tauri IPC overhead           |
| Memory Usage     | RAM     | <500MB | With 200-point history       |

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

| Component           | Replicas  | CPU (Request/Limit) | Memory (Request/Limit) | GPU       |
| ------------------- | --------- | ------------------- | ---------------------- | --------- |
| **nglab-api**       | 2-5       | 500m / 2000m        | 1Gi / 4Gi              | No        |
| **training-worker** | 1-4       | 2000m / 8000m       | 4Gi / 32Gi             | Yes (1-2) |
| **postgresql**      | 1 (HA: 3) | 500m / 2000m        | 1Gi / 4Gi              | No        |
| **redis**           | 1 (HA: 3) | 100m / 500m         | 256Mi / 1Gi            | No        |
| **prometheus**      | 1         | 500m / 1000m        | 2Gi / 4Gi              | No        |
| **grafana**         | 1         | 100m / 500m         | 256Mi / 512Mi          | No        |
| **jaeger**          | 1         | 500m / 1000m        | 1Gi / 2Gi              | No        |

### Kustomize Overlay Structure

```
infrastructure/global/k8s/
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
kubectl apply -k infrastructure/global/k8s/overlays/dev

# Deploy to staging
kubectl apply -k infrastructure/global/k8s/overlays/staging

# Deploy to production
kubectl apply -k infrastructure/global/k8s/overlays/prod

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

| Metric                   | Type      | Description              |
| ------------------------ | --------- | ------------------------ |
| `nglab_orders_total`     | Counter   | Total orders processed   |
| `nglab_order_latency_ms` | Histogram | Order processing latency |
| `nglab_portfolio_value`  | Gauge     | Current portfolio value  |
| `nglab_position`         | Gauge     | Current position size    |
| `nglab_training_loss`    | Gauge     | Current training loss    |
| `nglab_training_reward`  | Gauge     | Episode reward           |

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

| Table         | Purpose                         |
| ------------- | ------------------------------- |
| `experiments` | Experiment metadata             |
| `runs`        | Individual training runs        |
| `metrics`     | Training metrics (loss, reward) |
| `params`      | Hyperparameters                 |
| `artifacts`   | Model checkpoints               |

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

| Variable                      | Description            | Default          |
| ----------------------------- | ---------------------- | ---------------- |
| `NGLAB_LOG_LEVEL`             | Logging verbosity      | `INFO`           |
| `NGLAB_DATA_DIR`              | Data storage path      | `./data`         |
| `NGLAB_CACHE_DIR`             | Cache directory        | `./cache`        |
| `CUDA_VISIBLE_DEVICES`        | GPU selection          | All              |
| `WANDB_API_KEY`               | W&B authentication     | -                |
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
  "arena-update": ArenaUpdate;
  "trade-executed": TradeEvent;
  error: ErrorEvent;
  log: LogEvent;
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

## 21. Backend Internal Architecture

This section provides detailed diagrams of the Rust and Python backend internals, showing structs/classes, functions, and their relationships.

### 21.1 Rust Backend Architecture

#### Module Structure

```
rust/src/
├── lib.rs                      # PyO3 module entry (_nglab)
├── simulation/
│   ├── mod.rs
│   ├── orderbook.rs            # OrderBook, Order, PriceLevel, Trade
│   ├── gym.rs                  # TradingEnv, StepResult, ObservationBuffer
│   ├── polymarket.rs           # PolymarketArena, Market, Account
│   ├── multi_asset.rs          # MultiAssetEnv, AlgoOrder
│   └── risk.rs                 # RiskManager, RiskConfig, RiskStatus
├── errors/mod.rs               # ArenaError, ArenaResult
├── validation/mod.rs           # validate_price, validate_quantity
├── functions/math.rs           # SafeFloat, safe_div
├── models/                     # Black-Scholes, Rough Heston, etc.
├── moon/                       # ARIMA, GARCH, Exponential Smoothing
├── web/                        # PolymarketScraper, streaming
├── secret/vault.rs             # VaultManager, VaultEntry
├── utils/visualizer.rs         # RerunLogger
└── config.rs                   # Settings, configs
```

#### Core Structs & Relationships

```mermaid
classDiagram
    direction TB

    %% ===== SIMULATION CORE =====
    class TradingEnv {
        -orderbook: OrderBook
        -prices: Vec~f64~
        -current_step: usize
        -position: f64
        -cash: f64
        -initial_capital: f64
        -transaction_cost: f64
        -lookback: usize
        -risk_manager: RiskManager
        -obs_buffer: ObservationBuffer
        -rng: StdRng
        +new(capital, cost, lookback, max_steps, logging, seed) Self
        +load_prices(prices)
        +reset_rs() Vec~f64~
        +step_rs(action) Tuple
        +portfolio_value() f64
        +current_position() f64
        +risk_status() RiskStatus
    }

    class OrderBook {
        -bids: IndexMap~i64, PriceLevel~
        -asks: IndexMap~i64, PriceLevel~
        -stop_orders: Vec~Order~
        -next_order_id: u64
        -last_price: Option~f64~
        +new() Self
        +submit_limit_order(price, qty, side) Result
        +submit_market_order(qty, side) Result
        +submit_advanced_order(...) Result
        +check_triggers(price) Vec~Trade~
        +best_bid() Option~f64~
        +best_ask() Option~f64~
        +mid_price() Option~f64~
        +spread() Option~f64~
        +imbalance() f64
        +cancel_order(id) bool
        +modify_order(id, price, qty, ts) Option~u64~
        -match_order(order) Vec~Trade~
        -price_to_key(price) i64
    }

    class Order {
        +id: u64
        +price: f64
        +quantity: f64
        +filled: f64
        +side: Side
        +order_type: OrderType
        +timestamp: u64
        +trigger_price: Option~f64~
        +trailing_delta: Option~f64~
        +new(id, price, qty, side, type, ts) Self
        +new_advanced(...) Self
        +remaining() f64
        +is_filled() bool
        +is_iceberg() bool
        +refresh_iceberg()
    }

    class PriceLevel {
        +price: f64
        +orders: VecDeque~Order~
        +total_quantity: f64
        +new(price) Self
        +add_order(order)
        +remove_front() Option~Order~
        +is_empty() bool
    }

    class Trade {
        +maker_order_id: u64
        +taker_order_id: u64
        +price: f64
        +quantity: f64
        +side: Side
        +timestamp: u64
    }

    class RiskManager {
        -config: RiskConfig
        -returns_history: VecDeque~f64~
        -peak_value: f64
        -current_value: f64
        -daily_start_value: f64
        -status: RiskStatus
        +new(config, capital) Self
        +with_defaults(capital) Self
        +update(portfolio_value)
        +new_trading_day()
        +status() RiskStatus
        -calculate_metrics()
        -calculate_var() f64
    }

    class RiskConfig {
        +max_position_fraction: f64
        +daily_loss_limit: f64
        +max_drawdown: f64
        +var_confidence: f64
        +var_limit: f64
        +default() Self
    }

    class RiskStatus {
        +current_var: f64
        +current_drawdown: f64
        +daily_pnl: f64
        +daily_limit_breached: bool
        +drawdown_breached: bool
        +risk_score: u8
        +position_multiplier: f64
    }

    class ObservationBuffer {
        -data: Vec~f64~
        -shape: Tuple~usize, usize~
        +new(features, lookback) Self
        +update(row, values)
        +as_slice() Slice
        +reset()
        +to_vec() Vec~f64~
    }

    class StepResult {
        +observation: Vec~f64~
        +reward: f64
        +terminated: bool
        +truncated: bool
        +info: StepInfo
    }

    class StepInfo {
        +portfolio_value: f64
        +position: f64
        +cash: f64
        +sharpe_ratio: f64
        +total_steps: u64
    }

    %% ===== POLYMARKET =====
    class PolymarketArena {
        -markets: HashMap~String, Market~
        -price_history: HashMap~String, Vec~PriceTick~~
        -current_index: usize
        -account: Account
        -step: u64
        -taker_fee: f64
        +new(collateral, fee) Self
        +collateral() f64
        +current_step() u64
        +num_markets() usize
        +get_price(market_id) Option~f64~
        +get_position(market_id) Tuple
        +account_value() f64
        +realized_pnl() f64
        +load_markets(json)
        +load_price_history(market_id, csv)
        +buy_yes(market_id, amount) Result
        +buy_no(market_id, amount) Result
        +sell_yes(market_id, amount) Result
        +merge(market_id, amount) Result
        +split(market_id, amount) Result
        +advance() bool
        +reset(collateral)
    }

    class Market {
        +id: String
        +title: String
        +category: String
        +options: Vec~String~
        +resolved: bool
        +outcome: Option~String~
    }

    class Account {
        +collateral: f64
        +positions: HashMap~String, Tuple~
        +realized_pnl: f64
        +new(collateral) Self
        +get_position(market_id) Tuple
        +unrealized_pnl(prices) f64
    }

    class PriceTick {
        +timestamp: u64
        +price: f64
    }

    %% ===== MULTI-ASSET =====
    class MultiAssetEnv {
        -assets: Vec~String~
        -orderbooks: HashMap~String, OrderBook~
        -prices: HashMap~String, Vec~f64~~
        -current_step: usize
        -positions: HashMap~String, f64~
        -cash: f64
        -risk_manager: RiskManager
        -algo_orders: Vec~AlgoOrder~
        +new(assets, capital, cost, lookback, max_steps, seed) Self
        +load_prices(asset, prices)
        +portfolio_value() f64
        +reset_native(seed) Result
        +step_native(actions) Result
    }

    class AlgoOrder {
        +asset: String
        +side: Side
        +total_quantity: f64
        +remaining_quantity: f64
        +start_step: u64
        +end_step: u64
        +algo_type: AlgoType
    }

    %% ===== ENUMS =====
    class Side {
        <<enumeration>>
        Bid
        Ask
    }

    class OrderType {
        <<enumeration>>
        Limit
        Market
        StopLoss
        TakeProfit
        StopLimit
    }

    class AlgoType {
        <<enumeration>>
        TWAP
        VWAP
    }

    %% ===== ERROR HANDLING =====
    class ArenaError {
        <<enumeration>>
        OrderBook(String)
        InvalidOrder(String)
        InsufficientBalance
        InvalidPrice(f64)
        InvalidQuantity(f64)
        MarketNotFound(String)
        DataLoading(String)
        Python(String)
    }

    %% ===== RELATIONSHIPS =====
    TradingEnv *-- OrderBook : contains
    TradingEnv *-- RiskManager : contains
    TradingEnv *-- ObservationBuffer : contains
    TradingEnv ..> StepResult : returns
    TradingEnv ..> StepInfo : returns

    OrderBook *-- "many" PriceLevel : bids/asks
    OrderBook *-- "many" Order : stop_orders
    OrderBook ..> Trade : produces

    PriceLevel *-- "many" Order : contains

    Order --> Side : has
    Order --> OrderType : has

    Trade --> Side : has

    RiskManager *-- RiskConfig : uses
    RiskManager *-- RiskStatus : maintains

    PolymarketArena *-- "many" Market : contains
    PolymarketArena *-- Account : contains
    PolymarketArena *-- "many" PriceTick : price_history

    Account ..> Market : positions in

    MultiAssetEnv *-- "many" OrderBook : per asset
    MultiAssetEnv *-- RiskManager : contains
    MultiAssetEnv *-- "many" AlgoOrder : pending

    AlgoOrder --> Side : has
    AlgoOrder --> AlgoType : has
```

#### PyO3 Module Bindings

```mermaid
flowchart TB
    subgraph "Rust (_nglab module)"
        direction TB
        lib[lib.rs<br/>PyO3 Module Init]

        subgraph "Exposed Classes"
            Arena["Arena<br/>#[pyclass]"]
            TradingEnv_py["TradingEnv<br/>#[pyclass]"]
            OrderBook_py["OrderBook<br/>#[pyclass]"]
            PolymarketArena_py["PolymarketArena<br/>#[pyclass]"]
            MultiAssetEnv_py["MultiAssetEnv<br/>#[pyclass]"]
            RiskConfig_py["RiskConfig<br/>#[pyclass]"]
            RiskStatus_py["RiskStatus<br/>#[pyclass]"]
        end

        lib --> Arena
        lib --> TradingEnv_py
        lib --> OrderBook_py
        lib --> PolymarketArena_py
        lib --> MultiAssetEnv_py
        lib --> RiskConfig_py
        lib --> RiskStatus_py
    end

    subgraph "Python Import"
        import["from nglab._nglab import<br/>TradingEnv, OrderBook,<br/>PolymarketArena, Arena"]
    end

    TradingEnv_py -.->|"PyO3 FFI"| import
    OrderBook_py -.->|"PyO3 FFI"| import
    PolymarketArena_py -.->|"PyO3 FFI"| import
    Arena -.->|"PyO3 FFI"| import
```

#### Key Function Call Graph (TradingEnv)

```mermaid
flowchart TB
    subgraph "TradingEnv::step_rs(action)"
        step[step_rs]
        exec[execute_action]
        gen_obs[generate_observation_data]
        calc_reward[calculate_reward]
        calc_sharpe[calculate_sharpe]

        step --> exec
        step --> gen_obs
        step --> calc_reward
        calc_reward --> calc_sharpe
    end

    subgraph "execute_action"
        cur_price[current_price]
        ob_market[orderbook.submit_market_order]
        risk_update[risk_manager.update]

        exec --> cur_price
        exec --> ob_market
        exec --> risk_update
    end

    subgraph "OrderBook::submit_market_order"
        match[match_order]
        check_triggers[check_triggers]

        ob_market --> match
        match --> check_triggers
    end

    subgraph "generate_observation_data"
        ob_best[orderbook.best_bid/ask]
        ob_imb[orderbook.imbalance]

        gen_obs --> ob_best
        gen_obs --> ob_imb
    end
```

---

### 21.2 Python Backend Architecture

#### Module Structure

```
python/src/
├── main.py                          # Hydra entry point
├── cli/                             # Subcommand routing & args
├── configs/                         # Domain-organized dataclass configs
├── envs/                            # Gym environments & factory
├── models/                          # Multi-model library
│   ├── deep/                        # DL architectures
│   └── mac/                         # Classical ML
├── policies/                        # Action selection logic
├── pipeline/                        # Training loops & Lightning modules
├── backtesting/                     # Backtest engine & strategies
├── data/                            # Dataloaders & dataset definitions
├── api/                             # Inference & Health endpoints
├── db/                              # DB models & caching
├── constants/                       # Domain constants
├── exceptions.py                    # Centralized error types
└── utils/                           # Profiling, configuration, registries
```

#### Core Classes & Relationships

```mermaid
classDiagram
    direction TB

    %% ===== ENVIRONMENTS =====
    class TradingEnv_Py {
        <<gym.Env>>
        -_rust_env: _nglab.TradingEnv
        -observation_space: Box
        -action_space: Discrete
        +__init__(lookback, max_steps, feature_dim)
        +reset(seed, options) Tuple
        +step(action) Tuple
        +render()
        +close()
        -_get_observation() ndarray
        -_execute_action(action) float
        -_portfolio_value() float
        -_calculate_sharpe() float
    }

    class ClobEnv {
        <<TradingEnv>>
        -_orderbook: _nglab.OrderBook
        +__init__(...)
        +step(action) Tuple
    }

    class PolymarketEnv {
        <<gym.Env>>
        -_arena: _nglab.PolymarketArena
        -market_ids: List~str~
        -observation_space: Box
        -action_space: MultiDiscrete
        +__init__(market_ids, collateral, fee)
        +reset(seed, options) Tuple
        +step(action) Tuple
        -_get_observation() ndarray
        -_execute_market_action(mkt, act) float
        -_account_value() float
    }

    class VectorizedTradingEnv {
        -envs: List~TradingEnv~
        -num_envs: int
        +__init__(num_envs, capital, cost, lookback, max_steps, multiprocessing)
        +reset(seed, options) ndarray
        +step(actions) Tuple
        +step_async(actions)
        +step_wait() Tuple
        +load_prices(prices)
    }

    class TradingEnvWrapper {
        <<GymWrapper>>
        -_env: TradingEnv
        -device: torch.device
        +__init__(env, device, num_envs)
        -_make_specs(env, batch_size)
    }

    %% ===== POLICIES =====
    class Policy {
        <<ABC>>
        +act(observation)* Any
        +__call__(observation) Any
        +reset()
    }

    class NeuralPolicy {
        -model: nn.Module
        -cfg: Config
        +__init__(model, cfg)
        +act(observation) int
    }

    class ThresholdPolicy {
        -threshold: float
        +__init__(threshold)
        +act(observation) int
    }

    class BlackScholesPolicy {
        -strike: float
        -rate: float
        +__init__(strike, rate)
        +act(observation) int
    }

    %% ===== MODELS =====
    class TimeSeriesBackbone {
        <<nn.Module>>
        -encoder: nn.Module
        -model_name: str
        +__init__(cfg)
        +forward(x) Tensor
    }

    class create_deep_model {
        <<function>>
        +__call__(model_name, cfg) nn.Module
    }

    class create_mac_model {
        <<function>>
        +__call__(model_name, cfg) BaseEstimator
    }

    %% ===== LIGHTNING MODULES =====
    class BaseModule {
        <<LightningModule>>
        #model: nn.Module
        #cfg: Config
        +__init__(model, cfg)
        +configure_optimizers()
    }

    class RLLightningModule {
        -agent_module: TensorDictModule
        -critic_network: nn.Module
        -collector: SyncDataCollector
        -replay_buffer: ReplayBuffer
        -loss_module: ClipPPOLoss
        +__init__(cfg)
        +training_step(batch, idx) Tensor
        +validation_step(batch, idx)
        +on_train_epoch_end()
    }

    class SLLightningModule {
        -criterion: nn.Module
        +__init__(model, cfg, criterion)
        +training_step(batch, idx) Tensor
        +validation_step(batch, idx)
    }

    class VAELightningModule {
        -vae: VAE
        -beta: float
        +__init__(vae, cfg)
        +training_step(batch, idx) Tensor
        +_elbo_loss(x, recon, mu, logvar) Tensor
    }

    %% ===== BACKTESTING =====
    class BacktestEngine {
        -arena: _nglab.PolymarketArena
        -strategy: BaseStrategy
        -market_ids: List~str~
        -history: List~dict~
        +__init__(collateral, fee)
        +set_strategy(strategy)
        +load_data(markets_json, price_histories)
        +run() List~dict~
        +buy_yes(market_id, amount) float
        +buy_no(market_id, amount) float
        +sell_yes(market_id, amount) float
        +merge(market_id, amount) float
        +split(market_id, amount) float
    }

    class BaseStrategy {
        <<ABC>>
        #name: str
        #engine: BacktestEngine
        +__init__(name)
        +set_engine(engine)
        +on_market_data(market_id, price, ts)*
        +on_fill(market_id, amount, price, side)
    }

    class SMACrossoverStrategy {
        -short_window: int
        -long_window: int
        -prices: Dict
        +__init__(short, long)
        +on_market_data(market_id, price, ts)
    }

    class calculate_metrics {
        <<function>>
        +__call__(history, rf_rate) Dict
    }

    %% ===== DATA =====
    class FinancialDataset {
        <<TimeSeriesDataset>>
        -data: DataFrame
        -seq_len: int
        -pred_len: int
        -scaler: Scaler
        +__init__(csv_path, target, seq_len, pred_len, normalize, indicators)
        +__getitem__(idx) Tuple
        +__len__() int
    }

    class PolymarketDataset {
        <<Dataset>>
        -data: ndarray
        -seq_len: int
        -pred_len: int
        +__init__(name, dir, seq_len, pred_len)
        -_load_multivariate_data()
        +__getitem__(idx) Tuple
    }

    class create_dataloader {
        <<function>>
        +__call__(...) Tuple~DataLoader~
    }

    %% ===== API =====
    class BatchInferenceHandler {
        -model: nn.Module
        -batch_queue: Queue
        -cache: Redis
        +__init__(model, batch_size, timeout)
        +queue_inference(request) Future
        +process_batch()
    }

    %% ===== RELATIONSHIPS =====
    TradingEnv_Py --|> gym.Env
    ClobEnv --|> TradingEnv_Py
    PolymarketEnv --|> gym.Env

    VectorizedTradingEnv *-- "many" TradingEnv_Py : contains
    TradingEnvWrapper o-- TradingEnv_Py : wraps

    Policy <|-- NeuralPolicy
    Policy <|-- ThresholdPolicy
    Policy <|-- BlackScholesPolicy

    NeuralPolicy o-- TimeSeriesBackbone : uses

    TimeSeriesBackbone ..> create_deep_model : uses
    TimeSeriesBackbone ..> create_mac_model : uses

    BaseModule <|-- RLLightningModule
    BaseModule <|-- SLLightningModule
    BaseModule <|-- VAELightningModule

    RLLightningModule o-- TradingEnvWrapper : trains on
    RLLightningModule o-- TimeSeriesBackbone : policy network

    SLLightningModule o-- TimeSeriesBackbone : model
    SLLightningModule o-- FinancialDataset : data

    BacktestEngine o-- BaseStrategy : uses
    BaseStrategy <|-- SMACrossoverStrategy

    FinancialDataset --|> TimeSeriesDataset
    PolymarketDataset --|> Dataset

    BatchInferenceHandler o-- TimeSeriesBackbone : serves
```

---

### 21.3 Rust-Python Integration Architecture

#### PyO3 Bridge Overview

```mermaid
flowchart TB
    subgraph "Python Layer"
        direction TB

        subgraph "High-Level Wrappers"
            TradingEnvPy["TradingEnv<br/>(python/src/env/envs.py)"]
            PolymarketEnvPy["PolymarketEnv<br/>(python/src/env/envs.py)"]
            BacktestEng["BacktestEngine<br/>(python/src/backtesting/engine.py)"]
        end

        subgraph "TorchRL Integration"
            EnvWrapper["TradingEnvWrapper<br/>(env_wrapper.py)"]
            Collector["SyncDataCollector"]
            ReplayBuf["ReplayBuffer"]
        end

        subgraph "Training Pipeline"
            RLModule["RLLightningModule"]
            SLModule["SLLightningModule"]
            Trainer["Lightning Trainer"]
        end

        subgraph "Models"
            Backbone["TimeSeriesBackbone"]
            NPolicy["NeuralPolicy"]
        end
    end

    subgraph "PyO3 FFI Layer"
        direction LR
        style PyO3 fill:#f9f,stroke:#333

        TradingEnvRust["_nglab.TradingEnv"]
        OrderBookRust["_nglab.OrderBook"]
        PolymarketRust["_nglab.PolymarketArena"]
        ArenaRust["_nglab.Arena"]

        note1["Zero-copy NumPy arrays<br/>via PyArray"]
    end

    subgraph "Rust Layer"
        direction TB

        subgraph "simulation/"
            GymRs["gym.rs<br/>TradingEnv"]
            OrderBookRs["orderbook.rs<br/>OrderBook"]
            PolyRs["polymarket.rs<br/>PolymarketArena"]
            RiskRs["risk.rs<br/>RiskManager"]
        end

        subgraph "Support"
            Validation["validation/"]
            Errors["errors/"]
            Config["config.rs"]
        end
    end

    %% Python → PyO3
    TradingEnvPy -->|"self._rust_env"| TradingEnvRust
    PolymarketEnvPy -->|"self._arena"| PolymarketRust
    BacktestEng -->|"self.arena"| PolymarketRust

    %% PyO3 → Rust
    TradingEnvRust -.->|"#[pyclass]"| GymRs
    OrderBookRust -.->|"#[pyclass]"| OrderBookRs
    PolymarketRust -.->|"#[pyclass]"| PolyRs

    %% Rust internal
    GymRs --> OrderBookRs
    GymRs --> RiskRs
    PolyRs --> Errors

    %% Python training flow
    EnvWrapper --> TradingEnvPy
    Collector --> EnvWrapper
    RLModule --> Collector
    RLModule --> ReplayBuf
    RLModule --> Backbone
    NPolicy --> Backbone
    Trainer --> RLModule
```

#### Data Flow: Training Step

```mermaid
sequenceDiagram
    participant Trainer as Lightning Trainer
    participant RL as RLLightningModule
    participant Collector as SyncDataCollector
    participant Wrapper as TradingEnvWrapper
    participant PyEnv as TradingEnv (Python)
    participant PyO3 as PyO3 FFI
    participant RustEnv as TradingEnv (Rust)
    participant OB as OrderBook (Rust)
    participant Risk as RiskManager (Rust)

    Trainer->>RL: training_step(batch)
    RL->>Collector: rollout()

    loop For each step
        Collector->>Wrapper: step(action)
        Wrapper->>PyEnv: step(action)
        PyEnv->>PyO3: self._rust_env.step(action)

        PyO3->>RustEnv: step_rs(action)
        RustEnv->>RustEnv: execute_action(action)
        RustEnv->>OB: submit_market_order()
        OB->>OB: match_order()
        OB-->>RustEnv: Vec<Trade>
        RustEnv->>Risk: update(portfolio_value)
        Risk-->>RustEnv: RiskStatus
        RustEnv->>RustEnv: generate_observation_data()
        RustEnv->>RustEnv: calculate_reward()

        RustEnv-->>PyO3: (obs_array, reward, term, trunc, info)
        Note over PyO3: Zero-copy PyArray transfer
        PyO3-->>PyEnv: (np.ndarray, float, bool, bool, dict)
        PyEnv-->>Wrapper: TensorDict
        Wrapper-->>Collector: TensorDict
    end

    Collector-->>RL: Batch of trajectories
    RL->>RL: loss_module.forward(batch)
    RL->>RL: optimizer.step()
    RL-->>Trainer: loss
```

#### Data Flow: Backtesting

```mermaid
sequenceDiagram
    participant User as User Code
    participant Engine as BacktestEngine
    participant Strategy as BaseStrategy
    participant PyO3 as PyO3 FFI
    participant Arena as PolymarketArena (Rust)
    participant Account as Account (Rust)

    User->>Engine: __init__(collateral, fee)
    Engine->>PyO3: PolymarketArena(collateral, fee)
    PyO3->>Arena: new(collateral, fee)
    Arena-->>PyO3: arena instance
    PyO3-->>Engine: self.arena

    User->>Engine: load_data(markets_json, price_histories)
    Engine->>PyO3: arena.load_markets(json)
    PyO3->>Arena: load_markets_py(json)

    loop For each market
        Engine->>PyO3: arena.load_price_history(mkt_id, csv)
        PyO3->>Arena: load_price_history_py(mkt_id, csv)
    end

    User->>Engine: set_strategy(strategy)
    Engine->>Strategy: set_engine(self)

    User->>Engine: run()

    loop While arena.advance()
        Engine->>PyO3: arena.advance()
        PyO3->>Arena: advance_py()
        Arena-->>PyO3: bool (has more data)

        loop For each market
            Engine->>PyO3: arena.get_price(mkt_id)
            PyO3->>Arena: get_price_py(mkt_id)
            Arena-->>PyO3: Option<f64>

            Engine->>Strategy: on_market_data(mkt_id, price, ts)

            alt Strategy decides to trade
                Strategy->>Engine: buy_yes(mkt_id, amount)
                Engine->>PyO3: arena.buy_yes(mkt_id, amount)
                PyO3->>Arena: buy_yes_py(mkt_id, amount)
                Arena->>Account: update position
                Arena-->>PyO3: shares_received
                PyO3-->>Engine: float
                Engine->>Strategy: on_fill(mkt_id, shares, price, "buy")
            end
        end

        Engine->>Engine: record history snapshot
    end

    Engine-->>User: history list
```

---

### 21.4 File-Level Function/Struct Location Reference

#### Rust Files

| File                                        | Structs/Enums                                                             | Key Functions/Methods                                                                                                                                                              |
| ------------------------------------------- | ------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `rust/src/lib.rs:81-130`                    | `Arena`                                                                   | `new()`, `step_count()`, `new_py()`, `step_count_py()`                                                                                                                             |
| `rust/src/simulation/gym.rs:163-199`        | `TradingEnv`, `StepResult`, `StepInfo`, `ObservationBuffer`, `ActionType` | `new()`, `load_prices()`, `reset_rs()`, `step_rs()`, `portfolio_value()`, `execute_action()`, `calculate_reward()`, `calculate_sharpe()`                                           |
| `rust/src/simulation/orderbook.rs:31-887`   | `OrderBook`, `Order`, `PriceLevel`, `Trade`, `Side`, `OrderType`          | `submit_limit_order()`, `submit_market_order()`, `match_order()`, `check_triggers()`, `best_bid()`, `best_ask()`, `mid_price()`, `imbalance()`, `cancel_order()`, `modify_order()` |
| `rust/src/simulation/polymarket.rs:20-477`  | `PolymarketArena`, `Market`, `Account`, `PriceTick`                       | `load_markets()`, `load_price_history()`, `buy_yes()`, `buy_no()`, `sell_yes()`, `merge()`, `split()`, `advance()`, `reset()`, `account_value()`                                   |
| `rust/src/simulation/multi_asset.rs:17-175` | `MultiAssetEnv`, `AlgoOrder`, `MultiAssetStepResult`, `AlgoType`          | `new()`, `load_prices()`, `portfolio_value()`, `reset_native()`, `step_native()`                                                                                                   |
| `rust/src/simulation/risk.rs:15-200`        | `RiskManager`, `RiskConfig`, `RiskStatus`                                 | `new()`, `with_defaults()`, `update()`, `new_trading_day()`, `calculate_metrics()`, `calculate_var()`                                                                              |
| `rust/src/errors/mod.rs:13-82`              | `ArenaError`                                                              | (enum variants for error types)                                                                                                                                                    |
| `rust/src/validation/mod.rs:8-72`           | —                                                                         | `validate_price()`, `validate_quantity()`, `validate_asset()`, `validate_data_length()`, `validate_steps()`                                                                        |

#### Python Files

| File                                                           | Classes                                                            | Key Functions/Methods                                                                |
| -------------------------------------------------------------- | ------------------------------------------------------------------ | ------------------------------------------------------------------------------------ |
| `python/src/env/envs.py`                                       | `TradingEnv`, `ClobEnv`, `PolymarketEnv`                           | `reset()`, `step()`, `_get_observation()`, `_execute_action()`, `_portfolio_value()` |
| `python/src/env/env_wrapper.py`                                | `TradingEnvWrapper`                                                | `__init__()`, `_make_specs()`                                                        |
| `python/src/env/vectorized_env.py`                             | `VectorizedTradingEnv`, `SubprocVecEnv`                            | `reset()`, `step()`, `step_async()`, `step_wait()`, `make_vec_env()`                 |
| `python/src/policies/base.py`                                  | `Policy`                                                           | `act()`, `__call__()`, `reset()`                                                     |
| `python/src/policies/neural.py`                                | `NeuralPolicy`                                                     | `__init__()`, `act()`                                                                |
| `python/src/models/time_series.py`                             | `TimeSeriesBackbone`                                               | `__init__()`, `forward()`                                                            |
| `python/src/models/deep_factory.py`                            | —                                                                  | `create_deep_model()`                                                                |
| `python/src/pipeline/core/lightning/reinforcement_learning.py` | `RLLightningModule`                                                | `training_step()`, `validation_step()`, `on_train_epoch_end()`                       |
| `python/src/pipeline/core/lightning/supervised_learning.py`    | `SLLightningModule`                                                | `training_step()`, `validation_step()`                                               |
| `python/src/backtesting/engine.py`                             | `BacktestEngine`                                                   | `set_strategy()`, `load_data()`, `run()`, `buy_yes()`, `buy_no()`, `sell_yes()`      |
| `python/src/backtesting/strategy.py`                           | `BaseStrategy`, `Strategy`                                         | `set_engine()`, `on_market_data()`, `on_fill()`                                      |
| `python/src/backtesting/metrics.py`                            | —                                                                  | `calculate_metrics()`                                                                |
| `python/src/data/dataloaders.py`                               | `FinancialDataset`                                                 | `__init__()`, `__getitem__()`, `create_dataloader()`                                 |
| `python/src/data/polymarket_dataset.py`                        | `PolymarketDataset`                                                | `__init__()`, `_load_multivariate_data()`, `__getitem__()`                           |
| `python/src/api/inference.py`                                  | `BatchInferenceHandler`, `PredictionRequest`, `PredictionResponse` | `queue_inference()`, `process_batch()`                                               |
| `python/src/main.py`                                           | —                                                                  | `main()` (Hydra entry point)                                                         |

---

### 21.5 Cross-Component Call Matrix

This matrix shows which Python components call which Rust components:

| Python Component             | Rust Component Called                | Method(s) Used                                                                                                                                        |
| ---------------------------- | ------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `TradingEnv` (envs.py)       | `_nglab.TradingEnv`                  | `reset()`, `step()`, `load_prices()`, `get_observation()`                                                                                             |
| `ClobEnv` (envs.py)          | `_nglab.OrderBook`                   | `best_bid()`, `best_ask()`, `imbalance()`                                                                                                             |
| `PolymarketEnv` (envs.py)    | `_nglab.PolymarketArena`             | `reset()`, `get_price()`, `get_position()`, `buy_yes()`, `buy_no()`, `sell_yes()`, `advance()`                                                        |
| `BacktestEngine` (engine.py) | `_nglab.PolymarketArena`             | `load_markets()`, `load_price_history()`, `buy_yes()`, `buy_no()`, `sell_yes()`, `merge()`, `split()`, `advance()`, `account_value()`, `collateral()` |
| `VectorizedTradingEnv`       | `_nglab.TradingEnv` (via TradingEnv) | All TradingEnv methods                                                                                                                                |

---

### 21.6 Summary Statistics

| Metric                      | Rust   | Python   |
| --------------------------- | ------ | -------- |
| Total Lines of Code         | ~7,859 | ~15,000+ |
| Main Modules                | 11     | 12       |
| Structs/Classes             | ~20+   | ~50+     |
| Public Functions/Methods    | 100+   | 200+     |
| PyO3 Exposed Classes        | 7      | —        |
| Deep Learning Architectures | —      | 30+      |
| Classical ML Models         | —      | 25+      |

---

## 22. System Requirements

### Minimum Requirements

| Component   | Minimum      | Recommended      | Notes                          |
| ----------- | ------------ | ---------------- | ------------------------------ |
| **CPU**     | 4 cores      | 8+ cores         | AMD64 architecture required    |
| **RAM**     | 8 GB         | 32 GB            | 16 GB for multi-agent training |
| **GPU**     | None         | NVIDIA RTX 3080+ | CUDA 11.8+ for training        |
| **Storage** | 20 GB SSD    | 100 GB NVMe      | Fast I/O for data loading      |
| **OS**      | Ubuntu 22.04 | Ubuntu 24.04     | Also macOS 13+, Windows 11     |

### Software Dependencies

| Software    | Version | Purpose                     |
| ----------- | ------- | --------------------------- |
| **Rust**    | 1.75+   | Simulation engine           |
| **Python**  | 3.11+   | ML pipeline                 |
| **Node.js** | 20+     | Frontend development        |
| **CUDA**    | 11.8+   | GPU acceleration (optional) |
| **Docker**  | 24+     | Containerization            |
| **kubectl** | 1.28+   | Kubernetes deployment       |

### Network Requirements

| Port | Service     | Protocol   | Notes            |
| ---- | ----------- | ---------- | ---------------- |
| 8000 | API Server  | HTTP/HTTPS | Main application |
| 5432 | PostgreSQL  | TCP        | Database         |
| 6379 | Redis       | TCP        | Cache            |
| 4317 | Jaeger OTLP | gRPC       | Tracing          |
| 3000 | Grafana     | HTTP       | Monitoring UI    |
| 9090 | Prometheus  | HTTP       | Metrics          |

---

## Related Documentation

- [DEVELOPMENT.md](DEVELOPMENT.md) - Development setup guide
- [TESTING.md](TESTING.md) - Testing strategy
- [TUTORIAL.md](TUTORIAL.md) - Developer encyclopedia
- [AGENTS.md](AGENTS.md) - Agent taxonomy and policies
- [CONTRIBUTING.md](../git/CONTRIBUTING.md) - Contribution guidelines
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Common issues and fixes
- [ROADMAP.md](../moon/ROADMAP.md) - Development roadmap
- [README.md](../README.md) - Getting started

---

## Changelog

### Version 2.4.0 (Current)

- Added comprehensive Backend Internal Architecture section (Section 21)
  - Detailed Rust module structure and struct relationships
  - Complete class diagrams for all core Rust structs
  - PyO3 module bindings diagram
  - Function call graph for TradingEnv
  - Complete Python module structure and class relationships
  - Class diagrams for Python environments, policies, models, and pipeline
  - Rust-Python integration architecture with PyO3 bridge overview
  - Sequence diagrams for training step and backtesting data flows
  - File-level function/struct location reference tables
  - Cross-component call matrix

### Version 2.3.0

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

**Last Updated:** 2026-01-22
**Version:** 2.4.0 (The Complete System Blueprint)
**Maintainer:** NGLab Team
