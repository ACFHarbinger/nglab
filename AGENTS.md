# NGLab Agents: The Intelligence Corps

> **Focus**: Decision-Making Entities, Policies, and Learning Strategies.

This document serves as the handbook for the "Pilots" of the NGLab platform. While `ARCHITECTURE.md` describes the ship, `AGENTS.md` describes the crew.

---

## 1. Core Intelligence & Architectural Exposition

NGLab is a sophisticated **Multimodal Deep Reinforcement Learning (DRL)** platform engineered for high-frequency financial trading, multi-asset simulation, and prediction market analysis. It integrates high-performance systems programming (Rust), cutting-edge machine learning research (Python/PyTorch), and real-time interactive visualization (TypeScript/Tauri).

---

## 2. 🏗️ System Architecture Reuse

The platform is built on a decoupled, three-tier architecture that optimizes for performance, research flexibility, and operator efficiency.

```mermaid
graph TD
    subgraph "UI Layer (TypeScript/Tauri)"
        A[Dashboard Overview] --> B[Trading Terminal]
        B --> C[Market Analysis]
        C --> D[Model Control]
    end

    subgraph "intelligence Layer (Python/PyTorch)"
        E[Gymnasium Wrapper] --> F[Model Factory]
        F --> G[Policy Hub]
        G --> H[Training Pipeline]
    end

    subgraph "Simulation Layer (Rust Core)"
        I[Arena Engine] --> J[Order Book CLOB]
        J --> K[Polymarket Sim]
        K --> L[Model Suite]
    end

    D -- "Tauri Commands" --> I
    I -- "Event Stream" --> A
    E -- "PyO3 / Zero-Copy" --> I
    G -- "Inference" --> I
```

### The Intelligence Layer (Python)

The **Intelligence Layer** hosts an extensive library of deep learning architectures and training utilities.

- **The Model Factory**: 30+ implemented architectures, including:
  - **Sequencing**: LSTM, GRU, xLSTM, Mamba (SSM), NSTransformer.
  - **Generative**: VAE (with multimodal encoders), TimeGAN, Diffusion U-Net (1D).
  - **Advanced**: Neural ODEs (NODE), Physics-Informed Neural Networks (PINN), Differentiable Neural Computers (DNC), Spiking Neural Networks (SNN).
- **Policy Framework**: Integration with TorchRL for PPO and SAC agents, alongside classical threshold and quantitative policies.
- **Evolutionary HPO**: Automated hyperparameter optimization using Optuna for model refinement.

---

## 3. The Agent Taxonomy

NGLab employs a diverse set of agents, ranges from simple hard-coded heuristics to massive, pre-trained transformer policies.

### 3.1 Reinforcement Learning (RL) Agents
These agents learn via trial-and-error, optimizing the scalar reward signal $R_t$ provided by the environment.

#### **PPO (Proximal Policy Optimization)**
- **Role**: The "Steady Hand". Our default agent for stable, on-policy learning.
- **Architecture**: Actor-Critic.
    - **Actor**: $\pi(a|s)$. Outputs a probability distribution (Categorical for discrete, Gaussian for continuous) over actions.
    - **Critic**: $V(s)$. Estimates the expected future discounted reward.
- **Key Feature**: The "Clipped Objective" prevents the policy from updating too drastically in a single step, ensuring training stability during market regime shifts.

#### **SAC (Soft Actor-Critic)**
- **Role**: The "Explorer". An off-policy algorithm optimized for maximum entropy.
- **Objective**: Maximize $ \mathbb{E}[R_t] + \alpha \mathcal{H}(\pi(\cdot|s_t)) $.
- **Why it matters**: In financial markets, multiple actions might be equally valid. SAC encourages the agent to keep its options open (high entropy) rather than collapsing to a single deterministic strategy too early.

### 3.2 Heuristic (Rule-Based) Agents
These agents do not "learn" but execute pre-defined logic. They serve as essential baselines to benchmark RL performance.

#### **The "Market Maker"**
- **Logic**: Places limits orders at `BestBid - Spread` and `BestAsk + Spread`.
- **Goal**: Capture the bid-ask spread while remaining delta-neutral.
- **Inventory Control**: As inventory deviates from 0, it skews its quotes to encourage trades that flatten the position.

#### **The "Trend Follower"**
- **Logic**: Computes EMA(Short) and EMA(Long).
    - Buy if `EMA(Short) > EMA(Long)` (Golden Cross).
    - Sell if `EMA(Short) < EMA(Long)` (Death Cross).
- **Goal**: Capture massive unidirectional moves (gamma).

---

## 4. Policy Architectures

The "Brain" of the agent. We support swappable backbones.

### 4.1 The "TSMamba" Backbone
Our flagship architecture for time-series encoding.
- **Input**: A window of price updates `(Batch, SeqLen, Features)`.
- **Mechanism**: A Selective State Space Model (SSM). It compresses the history into a fixed-size latent state $h_t$ that evolves linearly.
- **Advantage**: $O(N)$ inference speed (vs $O(N^2)$ for Transformers), crucial for HFT latencies.

### 4.2 The "Visual" Backbone (CNN)
Used when the agent "sees" the Order Book as a 2D image (Price Level $\times$ Time).
- **Layers**: 1D Convolutions over the Price Level dimension.
- **Feature Extraction**: Detects clusters of liquidity (e.g., "walls") and order imbalances.

---

## 5. The Observation Space

What does the agent actually "see"?

### 5.1 Market Microstructure
- **LOB Snapshot**: The top 20 levels of Bids and Asks.
- **Imbalance**: $\frac{\text{Vol}_{bid} - \text{Vol}_{ask}}{\text{Vol}_{bid} + \text{Vol}_{ask}}$. A positive value implies buy pressure.
- **Spread**: $\text{Ask}_0 - \text{Bid}_0$. Narrow spreads imply high liquidity.

### 5.2 Portfolio State
- **Cash**: Available USD.
- **Inventory**: Current net position (signed).
- **Unrealized PnL**: Current profit/loss if flattened immediately.

### 5.3 Derived Signals
- **Volatility**: Rolling standard deviation of returns.
- **RSI**: Relative Strength Index (Momentum).
- **MACD**: Moving Average Convergence Divergence (Trend).

---

## 6. Interaction Lifecycle

How an Agent survives in the Arena:

1.  **Sense**: The environment wrapper collects raw data from Rust (`OrderBook`, `TradeHistory`).
2.  **Process**: The Observation Buffer normalizes these values (z-score) and stacks them into a tensor.
3.  **Think**: The Policy Network ($\pi_\theta$) performs a forward pass.
    - *Inference Mode*: Returns the mode of the distribution (Deterministic).
    - *Training Mode*: Samples from the distribution (Stochastic).
4.  **Act**: The chosen action index is sent to the Rust engine.
5.  **Learn**:
    - The `(State, Action, Reward, NextState)` tuple is pushed to a Replay Buffer.
    - Periodically, a Gradient Descent step updates $\theta$.

---

## 7. ⚡ Performance Specification

| Component                | Metric     | Achievement                            |
| :----------------------- | :--------- | :------------------------------------- |
| **Rust Matching Engine** | Latency    | < 100μs per order                      |
| **Environment Step**     | Speed      | > 20,000 steps/sec                     |
| **Data Bridge**          | Strategy   | Zero-copy NumPy integration via PyO3   |
| **Frontend UI**          | Framerate  | Locked 60 FPS with real-time streaming |
| **Memory Footprint**     | Efficiency | Core simulation < 50MB RSS             |

---

## 8. 📡 Data Ingestion & Scrapers

NGLab features specialized async scrapers built in Rust to ingest high-fidelity data from live markets:

- **Polymarket Scraper**: Multi-frequency ingestion (minutely/hourly/daily) for prediction market research.
- **WebSocket Hub**: Managed streaming for real-time price discovery and order flow.

---

## 9. 🛠️ Development & Stewardship

The project maintains professional software standards across all languages:

- **Testing**: Comprehensive suites across Rust (Criterion benchmarks), Python (Pytest), and TypeScript.
- **Documentation**: 100% docstring/JSDoc coverage; verified with compliance checkers.
- **CI/CD**: Automated linting, type-checking, and build validation for every commit.

---

## 10. 🚀 Future Roadmap for Agents

- **Multi-Agent Simulation**: Modeling competing RL agents in a single arena.
- **Distributed Training**: Scaling model learning across GPU clusters using Ray.
- **Microservices Shift**: Transitioning core components to specialized services for cloud scalability.
- **Causal Analysis**: Integrating causal inference to understand market drivers beyond simple correlation.

---

**NGLab** is built for the intersection of quantitative finance and cutting-edge AI research.
