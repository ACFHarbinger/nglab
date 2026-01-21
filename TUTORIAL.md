# NGLab: The Definitive Developer Encyclopedia & Technical Tutorial

> **Version**: 2.1 (The "Omnibus" Edition)
> **Target Audience**: Core Engineers, Quants, Research Scientists

Welcome to the internal documentation of **Nothing Gambles Like A Bot (NGLab)**. This encyclopedia is designed to be the ultimate reference for any developer, researcher, or quantitative engineer working on this high-performance financial intelligence platform.

NGLab is a masterclass in modern software architecture, bridging the gap between low-level systems performance (Rust), high-level research agility (Python), and real-time user interactivity (TypeScript/Tauri).

---

## Table of Contents

1.  [The NGLab Philosophy](#1-the-nglab-philosophy)
2.  [High-Level Architecture & Communication](#2-high-level-architecture--communication)
3.  [Rust: The Engine Room (`/rust`)](#3-rust-the-engine-room-rust)
    -   [3.1 The Order Matching Engine](#31-the-order-matching-engine-simulationorderbookrs)
    -   [3.2 Gymnasium Environments](#32-gymnasium-environments-simulationgymrs)
    -   [3.3 Risk Management System](#33-risk-management-system-simulationriskrs)
    -   [3.4 Prediction Markets Core](#34-prediction-markets-core-simulationpolymarketrs)
    -   [3.5 Econometrics & Project Moon](#35-econometrics--project-moon-moon)
    -   [3.6 Quantitative Finance Models](#36-quantitative-finance-models-models)
4.  [Python: The Research Brain (`/python`)](#4-python-the-research-brain-python)
    -   [4.1 Differential Evolution Hyperband (DEHB)](#41-differential-evolution-hyperband-dehb-pipelinehpodehbpy)
    -   [4.2 Time-Series Mamba Architecture](#42-time-series-mamba-architecture-modelsdeeprecurrenttsmambapy)
    -   [4.3 The VAE Framework](#43-the-vae-framework-modelsdeepautoencodersvaepy)
    -   [4.4 Neural Architecture Library](#44-neural-architecture-library-modelsdeep)
    -   [4.5 Reinforcement Learning Pipeline](#45-reinforcement-learning-pipeline-pipeline)
5.  [TypeScript: The Command Center (`/typescript`)](#5-typescript-the-command-center-typescript)
    -   [5.1 Analytical Visualization](#51-analytical-visualization-componentsanalysistabtsx)
    -   [5.2 React Architecture](#52-react-architecture)
    -   [5.3 Real-Time Data Streaming](#53-real-time-data-streaming)
    -   [5.4 Tauri Event IPC](#54-tauri-event-ipc)
6.  [Cross-Language Orchestration](#6-cross-language-orchestration)
    -   [6.1 The PyO3 Bridge](#61-the-pyo3-bridge)
    -   [6.2 Zero-Copy Memory Sharing](#62-zero-copy-memory-sharing)
7.  [Algorithm Deep Dives](#7-algorithm-deep-dives)
    -   [7.1 Price-Time Priority FIFO](#71-price-time-priority-fifo)
    -   [7.2 VAE Reparameterization Trick](#72-vae-reparameterization-trick)
    -   [7.3 Non-Stationary Normalization](#73-non-stationary-normalization)
8.  [Development Life Cycle](#8-development-life-cycle)
    -   [8.1 Setup & Environment](#81-setup--environment)
    -   [8.2 Training your first Agent](#82-training-your-first-agent)
    -   [8.3 Developing new UI components](#83-developing-new-ui-components)
9.  [Extending NGLab](#9-extending-nglab)
    -   [9.1 Adding a New Neural Model](#91-adding-a-new-neural-model)
    -   [9.2 Adding a New Order Type](#92-adding-a-new-order-type)
10. [Infrastructure & Reliability](#10-infrastructure--reliability)
    -   [10.1 Monitoring & Tracing](#101-monitoring--tracing)
    -   [10.2 CI/CD Pipeline](#102-cicd-pipeline)
11. [Exhaustive Code Search & Reference](#11-exhaustive-code-search--reference)
12. [Glossary of Terms](#12-glossary-of-terms)

---

## 1. The NGLab Philosophy

Financial markets are adversarial, multimodal, and fundamentally non-stationary. Typical trading bots often fail because they are either too slow to react to microstructure changes or too simple to understand macro sentiment shifts. NGLab was built on three core pillars to address these failures:

### 1.1 Zero-Latency Simulation
In Reinforcement Learning (RL), the "Step" is the atom of time. If a simulation step takes 100ms, training an agent for 1 million steps takes 27 hours. If the step takes 100μs (as it does in our Rust core), that same training takes 1.6 minutes. Rust's strict memory management and lack of a Garbage Collector (GC) allow for deterministic, microsecond-level simulations.

### 1.2 Multi-Scale Modeling
We fuse three scales of data:
1.  **Microstructure**: Order Book depth, imbalance, and queue position (Source: Rust/CLOB).
2.  **Price Action**: OHLCV candlesticks and technical indicators (Source: Project Moon).
3.  **Global Alpha**: Sentiment clusters and news flow (Source: Python/Deep Learning).

### 1.3 Real-Time Observability
A black-box agent is a liability. NGLab's GUI doesn't just show a line chart; it shows a live bid-ask ladder, real-time risk scores (VaR), and "attention maps" of what the transformer models are looking at.

---

## 2. High-Level Architecture & Communication

NGLab uses a **Hybrid Polyglot Architecture**. Each language is chosen for its unique strengths.

### 2.1 Component Breakdown
- **Rust Core**: Owned by the simulation. It runs the "Physics".
- **Python Pipeline**: Owned by the researcher. It runs the "Decision Engine".
- **TypeScript GUI**: Owned by the operator. It runs the "Observation Layer".

### 2.2 Communication Protocols
1.  **Rust ⇄ Python**: Facilitated by **PyO3**. We use the `#[pyclass]` attribute on Rust structs to make them appear as native classes in Python.
2.  **Rust ⇄ TypeScript**: Facilitated by **Tauri IPC**. We emit JSON-serialized events over the global event bus.
3.  **Python ⇄ Data**: Facilitated by **Hydra**. A centralized configuration system that ensures all components share a single source of truth for hyperparameters.

---

## 3. Rust: The Engine Room (`/rust`)

The code in `/rust` is the most performance-critical part of the project. It aims for zero-allocation in the hot-path and leverages SIMD where applicable.

### 3.1 The Order Matching Engine (`simulation/orderbook.rs`)
The `OrderBook` is a highly optimized Central Limit Order Book (CLOB).

#### Core Data Structures:
- `IndexMap<i64, PriceLevel>`: We use fixed-point arithmetic (`price * 10,000`) to store prices as integers, avoiding floating-point precision issues during equality checks.
- `VecDeque<Order>`: Each price level maintains a FIFO queue of orders.

#### Key Functions:
- `match_order(order: Order) -> Vec<Trade>`: The heart of the matching logic. Matches volume until the order is filled or liquidity is exhausted.
- `check_triggers(current_price: f64) -> Vec<Trade>`: Periodically converts Stop-Loss/Take-Profit orders to Market orders.

### 3.2 Gymnasium Environments (`simulation/gym.rs`)
This module implements the `TradingEnv`, the direct bridge to Python's RL models via `gymnasium`.

#### The "Step" Lifecycle:
1.  **Action**: Agent sends an action index (0: Hold, 1: Buy, 2: Sell) or continuous vector.
2.  **Execution**: `OrderBook` calculates execution price/slippage via `execute_action`.
3.  **State Update**: `Portfolio` updates cash and positions.
4.  **Observation**: Returns a zero-copy numpy array slice of recent prices and indicators.
5.  **Reward**: Calculated as $ R_t = \text{Returns}_t - \text{Costs} - \text{DrawdownPenalty} $.

#### Zero-Copy Optimization
To avoid overhead, we use a pre-allocated `ObservationBuffer` (flat `Vec<f64>`) reshaped into a `(lookback, features)` matrix. Python accesses this memory directly via `numpy` pointers, bypassing Serde serialization entirely.

### 3.3 Risk Management System (`simulation/risk.rs`)
The `RiskManager` struct acts as the automated safety officer for the system. It implements active monitoring of drawdown, value-at-risk (VaR), and daily loss limits.

#### Configuration (`RiskConfig`)
- `max_position_fraction` (f64): Hard limit on single position size (default: 0.10).
- `daily_loss_limit` (f64): Maximum daily PnL drop before halting (default: 0.02).
- `var_confidence` (f64): Confidence level for VaR calc (default: 0.95).
- `var_lookback` (usize): Rolling window size for historical simulation (default: 252).

#### Value-at-Risk (VaR) Calculation
We use a **Historical Simulation** approach.
1.  Collect historical daily returns in a circular buffer (`VecDeque<f64>`).
2.  Sort returns: $ R_{sorted} = \text{sort}(R) $.
3.  Find the percentile index: $ i = \lfloor (1 - \text{conf}) \times N \rfloor $.
4.  $ \text{VaR} = -R_{sorted}[i] $.
This method makes no assumption about the normality of returns (unlike Parametric VaR), capturing "fat tails" typical in crypto/prediction markets.

### 3.4 Prediction Markets Core (`simulation/polymarket.rs`)
Implemented in `PolymarketArena`, this module handles the unique mechanics of Conditional Token Framework (CTF) markets found on Polymarket.

#### Conditional Token Mechanics
In a binary market, 1 unit of Collateral (USDC) can be **Split** into:
-   1 **YES** Token
-   1 **NO** Token
Conversely, 1 YES + 1 NO can be **Merged** back into 1 USDC.

#### NegRisk & Cross-Collateralization
The system tracks positions in a `HashMap<String, (f64, f64)>` (Yes/No tuples).
-   **realized_pnl**: Updated on selling.
-   **unrealized_pnl**: Calculated as `(Yes * Price_Yes) + (No * Price_No) - Cost_Basis`.

### 3.5 Econometrics & Project Moon (`moon/`)
Project Moon provides classical statistical priors to our deep learning models.

-   **`arima.rs`**: Implements $p, d, q$ parameters. It uses a Maximum Likelihood Estimation (MLE) approach to fit the AutoRegressive and Moving Average components.
-   **`garch.rs`**: Specifically models "Volatility Clustering"—the phenomenon where high volatility periods tend to be followed by high volatility.
-   **`prophet.rs`**: Handles additive trend seasonal components. It identifies changepoints (structural breaks) in price action using a linear trend model with adaptive changepoints.

### 3.6 Quantitative Finance Models (`models/`)
-   **`black_scholes.rs`**: Beyond just pricing, it provides the "Greeks":
    -   **Delta**: Sensitivity to price change ($ \frac{\partial V}{\partial S} $).
    -   **Gamma**: Rate of change of Delta ($ \frac{\partial^2 V}{\partial S^2} $).
    -   **Theta**: Time decay ($ \frac{\partial V}{\partial t} $).
    -   **Vega**: Volatility sensitivity ($ \frac{\partial V}{\partial \sigma} $).
-   **`rough_heston.rs`**: A breakthrough in volatility modeling that uses Fractional Brownian Motion (fBm) to capture the fractal nature of market volatility.

---

## 4. Python: The Research Brain (`/python`)

The Python layer is where the high-level logic resides, serving as the playground for experimental architectures.

### 4.1 Differential Evolution Hyperband (DEHB) (`pipeline/hpo/dehb.py`)
We implement a custom `DifferentialEvolutionHyperband` optimizer that combines evolutionary search with resource allocation.

#### Algorithm Logic
1.  **Population Initialization**: Start with a random population of configurations.
2.  **Successive Halving (SH)**: We use `SynchronousHalvingBracketManager` to manage "brackets".
    -   Configurations start at `min_fidelity` (e.g., 10 training epochs).
    -   Top performaning fraction ($1/\eta$) are promoted to higher fidelity.
3.  **Differential Evolution (DE)**: Instead of random sampling for new configurations, we use DE mutation:
    $$ v_i = x_{r1} + F \cdot (x_{r2} - x_{r3}) $$
    This allows the search to "evolve" towards better hyperparameters over time.
4.  **Async Execution**: We use `dask.distributed` (`Client`) to parallelize job submission across workers/GPUs. The `_submit_job` and `_fetch_results_from_workers` methods handle the async future resolution.

### 4.2 Time-Series Mamba Architecture (`models/deep/recurrent/tsmamba.py`)
Mamba is a Selective State Space Model (SSM) that offers linear scaling $O(N)$ with sequence length, unlike Transformers' $O(N^2)$.

#### `TSMamba` Structure
```python
class TSMamba(nn.Module):
    def __init__(...):
        # 1. Input Projection
        self.encoder = nn.Linear(input_dim, d_model)
        
        # 2. Stacked Mamba Blocks
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state=16, d_conv=4, expand=2)
            for _ in range(n_layers)
        ])
        
        # 3. Output Projection
        self.head = nn.Linear(d_model, output_dim * forecast_horizon)
```

### 4.3 The VAE Framework (`models/deep/autoencoders/vae.py`)
The Variational Autoencoder is used for **Market Regime Detection**. By compressing price windows into a low-dimensional latent space, we can cluster these points to see if the market is currently in a "Trend", "Range", or "Crash" state.

#### Architecture
-   **Encoder**: Maps `(Batch, Seq, Feat)` → `Latent Mean` ($\mu$) & `LogVariance` ($\sigma^2$).
-   **Reparameterization**: $z = \mu + \sigma \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, 1)$.
-   **Decoder**: Maps `z` → `Reconstruction`.
-   **Loss**: $ \mathcal{L} = \text{MSE}(x, \hat{x}) + D_{KL}(q(z|x) || p(z)) $.

### 4.4 Neural Architecture Library (`models/deep/`)
NGLab's model library is categorized by "Decision Paradigms".

-   **Temporal Convolutional Networks (TCN)**: Efficient 1D convolutions for detecting patterns like "Head and Shoulders" at different scales.
-   **NSTransformers**: Features "Stationary Flow", which de-stations the input data, passes it through attention, and then re-stations it for prediction. This is essential for prices which "drift" over time.

### 4.5 Reinforcement Learning Pipeline (`pipeline/`)
The `pipeline` directory manages the lifecycle of agent training.
-   **`train.py`**: The unified entry point. Supports PPO, SAC, and DQN.
-   **`distributed_train.py`**: Leverages `torch.distributed` for multi-GPU training.
-   **`env_wrapper.py`**: Wraps our Rust `TradingEnv` into a `TorchRL.EnvBase`, converting Rust structs into PyTorch Tensors efficiently.

---

## 5. TypeScript: The Command Center (`/typescript`)

The GUI provides "Situational Awareness" using React 19 and Tauri 2.0.

### 5.1 Analytical Visualization (`components/AnalysisTab.tsx`)
This giant component handles Exploratory Data Analysis (EDA).

#### CSV Parsing & Normalization
We use `PapaParse` to ingest CSV data. The `processFile` function:
1.  Detects timestamp columns (`Date`, `Time`, `Timestamp`).
2.  Normalizes to Unix ms.
3.  Identifies "candidate" series (numerical columns excluding timestamp).
4.  Auto-sorts candidates by last value magnitude.

#### Highcharts Modules
We dynamically load specialized modules to support:
-   **Heatmaps**: For 2D returns correlation matrices over time buckets.
-   **Candlesticks**: Using `Highcharts.Stock` for financial plotting.
-   **Indicators**: Built-in SMA, MACD, RSI overlays.

### 5.2 React Architecture
We follow a **Component-Based Widget** architecture. Each widget (Price Chart, Order Book, Risk Score) is an independent unit that subscribes to the `useArena` hook.

### 5.3 Real-Time Data Streaming
The `AnalysisTab` also supports a "Live Mode" for Polymarket data.
-   **Polling**: `setInterval` triggers every 1s.
-   **Sync**: Updates a `latestPricesRef` to avoid React closure staleness.
-   **History Stitching**: Fetches the last 100 points from the CLOB API (`/prices-history`) on INIT, then appends live socket updates locally.

### 5.4 Tauri Event IPC (`hooks/useArena.ts`)
Communication is handled via a pub-sub model.
1.  Rust backend emits `arena-update`.
2.  Tauri bridge serializes the Rust struct to JSON.
3.  The hook updates the internal React `data` state.
4.  Subscribed widgets re-render with the new data.

---

## 6. Cross-Language Orchestration

### 6.1 The PyO3 Bridge
In `rust/lib.rs`, we define the module that allows Python to import Rust classes directly:
```rust
#[pymodule]
fn _nglab(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Arena>()?;
    m.add_class::<TradingEnv>()?;
    Ok(())
}
```
This is compiled into a `.so` (Linux) or `.pyd` (Windows) file that Python can `import`.

### 6.2 Zero-Copy Memory Sharing
When Rust generates observations:
1.  Rust allocates memory.
2.  Creates a Python `ArrayObject` pointing to that address.
3.  Python reads without copying, reducing CPU overhead by ~30%.

---

## 7. Algorithm Deep Dives

### 7.1 Price-Time Priority FIFO (Rust)
*Trade Matching Logic in `orderbook.rs`:*
1.  **Ask** for 100 @ \$50.
2.  **Engine** checks Bids.
3.  **Match**: 50 @ \$50 (Order A, 10:00:01).
4.  **Match**: 50 @ \$50 (Order B, 10:00:02).
5.  Order A fills first due to time priority.
6.  If Order B was only for 30 shares, the remaining 20 shares of the Ask order are added to the Ask side.

### 7.2 VAE Reparameterization Trick (Python)
Allows backprop through random sampling:
$$ z = \mu + \sigma \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, 1) $$
Since $\mu$ and $\sigma$ are outputs of our network, we can compute $\frac{\partial z}{\partial \mu}$ and $\frac{\partial z}{\partial \sigma}$.

### 7.3 Non-Stationary Normalization (Python)
1.  Calculate window stats: $\mu_{win}, \sigma_{win}$.
2.  Normalize inputs: $x_{norm} = (x - \mu_{win}) / \sigma_{win}$.
3.  Predict: $y_{pred}$.
4.  De-normalize: $y = y_{pred} \cdot \sigma_{win} + \mu_{win}$.
This handles price drift effectively.

---

## 8. Development Life Cycle

### 8.1 Setup & Environment
Use `just` for rapid bootstrapping:
```bash
just setup  # Compiles Rust, installs Python deps, sets up Node
```

### 8.2 Training your first Agent
Once setup is complete, you can train a baseline PPO agent:
```bash
python python/src/main.py algorithm=ppo task=trading model=tsmamba
```
Logs will appear in `./outputs/` and can be visualized in TensorBoard.

### 8.3 Developing new UI components
```bash
npm run tauri dev
```

---

## 9. Extending NGLab

### 9.1 Adding a New Neural Model
1.  Create `python/src/models/deep/my_model.py`.
2.  Inherit form `nn.Module`.
3.  Register in `deep_factory.py`.
4.  Add config in `python/src/conf/model/`.

### 9.2 Adding a New Order Type
1.  Update `OrderType` enum in `rust/src/simulation/orderbook.rs`.
2.  Implement trigger logic in `match_order`.
3.  Add UI inputs in `TradingFormWidget.tsx`.

---

## 10. Infrastructure & Reliability

### 10.1 Monitoring & Tracing
We use **OpenTelemetry**. Each simulation step is tracked as a "Span". If the latency of the Python decision-making increases, it shows up as a bottleneck in the Jaeger UI.

### 10.2 CI/CD Pipeline
-   **Rust**: `cargo test`, `cargo clippy` (Strict).
-   **Python**: `pytest`, `mypy` (Strict typing).
-   **TypeScript**: `npm test` (Vitest).

---

## 11. Exhaustive Code Search & Reference

### High-Level File Index:

| File Name | Functionality | Language |
| :--- | :--- | :--- |
| `rust/src/simulation/orderbook.rs` | CLOB Matching Engine | Rust |
| `rust/src/simulation/gym.rs` | RL Environment Wrapper | Rust |
| `rust/src/simulation/risk.rs` | Risk Management & VaR | Rust |
| `rust/src/moon/arima.rs` | ARIMA Forecasting | Rust |
| `rust/src/models/black_scholes.rs` | Options pricing | Rust |
| `python/src/pipeline/train.py` | Training Loop Entry Point | Python |
| `python/src/pipeline/hpo/dehb.py` | Evolutionary Hyperparam Opt | Python |
| `python/src/models/deep/tsmamba.py` | Mamba Architecture | Python |
| `python/src/models/deep/autoencoders/vae.py` | Market Regime VAE | Python |
| `typescript/src/hooks/useArena.ts` | Data Sync Hook | TS |
| `typescript/src/components/AnalysisTab.tsx`| Analytics & Charts | TS |
| `typescript/src-tauri/src/lib.rs` | Tauri IPC Backend | Rust |

---

## 12. Advanced Topics

### 12.1 Distributed Training with Ray

For large-scale experiments, we support distributed training using Ray:

```python
import ray
from ray import tune
from nglab.pipeline import train_agent

ray.init()

analysis = tune.run(
    train_agent,
    config={
        "learning_rate": tune.loguniform(1e-5, 1e-2),
        "entropy_coef": tune.uniform(0.0, 0.1),
        "num_envs": tune.choice([8, 16, 32]),
    },
    num_samples=50,
    resources_per_trial={"cpu": 4, "gpu": 1},
)

best_config = analysis.best_config
```

### 12.2 Multi-Agent Reinforcement Learning

NGLab supports multi-agent scenarios where multiple agents compete in the same arena:

```python
from nglab.env import MultiAgentArena
from nglab.agents import PPOAgent, SACAgent

arena = MultiAgentArena(num_agents=4)
agents = [
    PPOAgent("market_maker_1"),
    PPOAgent("market_maker_2"),
    SACAgent("trend_follower"),
    RandomAgent("noise_trader"),
]

for episode in range(1000):
    obs = arena.reset()
    done = False
    while not done:
        actions = {agent.name: agent.act(obs[agent.name]) for agent in agents}
        obs, rewards, dones, infos = arena.step(actions)
        done = all(dones.values())
```

### 12.3 Custom Reward Shaping

Defining custom reward functions for specific trading objectives:

```python
from nglab.rewards import RewardFunction

class SharpeReward(RewardFunction):
    def __init__(self, window: int = 100):
        self.returns_buffer = deque(maxlen=window)
    
    def compute(self, portfolio_value: float, prev_value: float) -> float:
        ret = (portfolio_value - prev_value) / prev_value
        self.returns_buffer.append(ret)
        
        if len(self.returns_buffer) < 10:
            return ret
        
        mean_ret = np.mean(self.returns_buffer)
        std_ret = np.std(self.returns_buffer) + 1e-8
        return mean_ret / std_ret
```

### 12.4 Model Ensembling

Combining multiple models for robust predictions:

```python
from nglab.ensembles import EnsembleModel

models = [
    TSMamba(config1),
    NSTransformer(config2),
    LSTM(config3),
]

ensemble = EnsembleModel(
    models=models,
    aggregation="weighted_average",
    weights=[0.5, 0.3, 0.2],
)

prediction = ensemble.predict(observation)
```

### 12.5 Backtesting Framework

Running historical backtests with realistic market impact:

```python
from nglab.backtest import Backtester, SlippageModel

backtester = Backtester(
    data_path="data/btc_2023.csv",
    initial_capital=100000,
    slippage=SlippageModel(base_bps=1, volume_impact=0.1),
    transaction_cost=0.001,
)

results = backtester.run(agent)
print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
print(f"Max Drawdown: {results.max_drawdown:.2%}")
print(f"Total Return: {results.total_return:.2%}")
```

---

## 13. Debugging Guide

### 13.1 Common Issues and Solutions

| Issue | Symptom | Solution |
|-------|---------|----------|
| Agent not learning | Reward stays flat | Check observation normalization |
| Memory leak | RAM grows continuously | Clear replay buffer, use bounded buffers |
| NaN in loss | Training crashes | Add gradient clipping, check for div by zero |
| Slow steps | >10ms per step | Profile with `samply`, reduce Python overhead |
| GPU idle | 0% utilization | Increase batch size or num_envs |

### 13.2 Debugging Rust-Python Interactions

```python
# Enable detailed PyO3 logging
import os
os.environ["RUST_LOG"] = "debug"
os.environ["RUST_BACKTRACE"] = "1"

import nglab
```

### 13.3 Profiling the Training Loop

```python
import torch.profiler

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    profile_memory=True,
) as prof:
    for step in range(100):
        agent.train_step(batch)

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

---

## 14. API Reference

### 14.1 Core Rust Types

```rust
/// The main trading environment
pub struct TradingEnv {
    pub cash: f64,
    pub position: f64,
    pub portfolio_value: f64,
    // ...
}

impl TradingEnv {
    pub fn new(config: EnvConfig) -> Self;
    pub fn reset(&mut self, seed: Option<u64>) -> Observation;
    pub fn step(&mut self, action: Action) -> StepResult;
}
```

### 14.2 Python Model Interface

```python
class BaseModel(nn.Module):
    """Base class for all NGLab models."""
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        raise NotImplementedError
    
    def predict(self, x: Tensor) -> Tensor:
        """Inference mode prediction."""
        self.eval()
        with torch.no_grad():
            return self.forward(x)
```

### 14.3 TypeScript Hook Interfaces

```typescript
interface ArenaState {
  stepInfo: StepInfo;
  orderBook: OrderBook;
  priceHistory: number[];
  isRunning: boolean;
}

interface UseArenaResult {
  arenaState: ArenaState;
  start: () => Promise<void>;
  stop: () => Promise<void>;
  reset: () => Promise<void>;
}

function useArena(): UseArenaResult;
```

---

## 15. Glossary of Terms

| Term | Definition |
|------|------------|
| **Alpha** | Edge/Strategy generating excess returns |
| **Arena** | The global simulation container |
| **Backbone** | The neural network architecture (e.g., Mamba, Transformer) |
| **CLOB** | Central Limit Order Book |
| **CTF** | Conditional Token Framework (Splitting collateral into outcome tokens) |
| **DEHB** | Differential Evolution Hyperband (Optimization algorithm) |
| **Drift** | Price movement away from the mean |
| **Entropy** | Measure of policy randomness in RL |
| **Fidelity** | Resource level in multi-fidelity optimization |
| **Greeks** | Derivatives representing option risk (Delta, Gamma, Vega) |
| **Hydra** | Python configuration management |
| **Imbalance** | Ratio of bid to ask volume |
| **KL Divergence** | Measure of distribution difference in VAE |
| **Latent Space** | Compressed representation in autoencoders |
| **LOB** | Limit Order Book |
| **Mamba** | State Space Model architecture |
| **Maturin** | Rust-Python build tool |
| **Observation** | State vector seen by the agent |
| **Policy** | Mapping from states to actions |
| **Polymarket** | Prediction market using binary outcome tokens |
| **PyO3** | Rust-Python bindings |
| **Replay Buffer** | Storage for experience tuples in off-policy RL |
| **Reparameterization** | Trick enabling gradient flow through sampling |
| **Sharpe Ratio** | Risk-adjusted return metric |
| **Slippage** | Difference between expected and executed price |
| **SSM** | State Space Model |
| **Tauri** | Rust-based Electron alternative for the GUI |
| **TorchRL** | PyTorch-based RL library |
| **VaR** | Value at Risk (Maximum Probable Loss) |
| **Vectorized Env** | Multiple parallel environment instances |
| **Zero-Copy** | Memory sharing without copying data |

---

## 16. Changelog

### Version 2.1 (Current)
- Added Advanced Topics section with distributed training and multi-agent support
- Expanded API Reference with Rust, Python, and TypeScript interfaces
- Added comprehensive Debugging Guide
- Expanded Glossary to 30+ terms

### Version 2.0
- Initial "Omnibus Edition" with full cross-language documentation
- Deep dives into DEHB, Mamba, and VAE architectures
- Added Code Reference tables

### Version 1.0
- Basic tutorial structure
- High-level architecture overview

---

*This guide is the living foundation of NGLab. Happy Hacking!*
