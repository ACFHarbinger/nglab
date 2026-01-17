# NGLab - Next Gen Laboratory

## Core Intelligence & Architectural Exposition

NGLab is a sophisticated **Multimodal Deep Reinforcement Learning (DRL)** platform engineered for high-frequency financial trading, multi-asset simulation, and prediction market analysis. It integrates high-performance systems programming (Rust), cutting-edge machine learning research (Python/PyTorch), and real-time interactive visualization (TypeScript/Tauri).

---

## 🏗️ System Architecture: The Triad Design

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

### 1. The Simulation Engine (Rust)
The **Core Layer** provides deterministic, low-latency execution of market mechanics.
- **`TradingEnv`**: A Gymnasium-compatible environment optimized for RL agents, featuring zero-copy data transfer to Python.
- **`OrderBook`**: A standard Central Limit Order Book (CLOB) implementation with price-time priority matching (O(log n) efficiency).
- **`PolymarketArena`**: A specialized simulation for prediction markets, handling binary outcome tokens and AMM dynamics.
- **Quantitative Suite**: High-fidelity mathematical models including Black-Scholes, Rough Heston, and Rough Bergomi for synthetic price generation and risk management.

### 2. Research & Learning Pipeline (Python)
The **Intelligence Layer** hosts an extensive library of deep learning architectures and training utilities.
- **The Model Factory**: 30+ implemented architectures, including:
    - **Sequencing**: LSTM, GRU, xLSTM, Mamba (SSM), NSTransformer.
    - **Generative**: VAE (with multimodal encoders), TimeGAN, Diffusion U-Net (1D).
    - **Advanced**: Neural ODEs (NODE), Physics-Informed Neural Networks (PINN), Differentiable Neural Computers (DNC), Spiking Neural Networks (SNN).
- **Policy Framework**: Integration with TorchRL for PPO and SAC agents, alongside classical threshold and quantitative policies.
- **Evolutionary HPO**: Automated hyperparameter optimization using Optuna for model refinement.

### 3. Operator Interface (TypeScript/Tauri 2.0)
The **Interaction Layer** provides a "Bloomberg-tier" dashboard for real-time monitoring and control.
- **Live Terminal**: High-performance charting using `lightweight-charts` and real-time order book visualizations.
- **Command & Control**: Dedicated modules for scraper management, model deployment, and live simulation steering.
- **Streaming Pipeline**: WebSocket-driven event updates providing 60 FPS visual feedback on market dynamics.

---

## ⚡ Performance Specification

| Component | Metric | Achievement |
| :--- | :--- | :--- |
| **Rust Matching Engine** | Latency | < 100μs per order |
| **Environment Step** | Speed | > 20,000 steps/sec |
| **Data Bridge** | Strategy | Zero-copy NumPy integration via PyO3 |
| **Frontend UI** | Framerate | Locked 60 FPS with real-time streaming |
| **Memory Footprint** | Efficiency | Core simulation < 50MB RSS |

---

## 📡 Data Ingestion & Scrapers

NGLab features specialized async scrapers built in Rust to ingest high-fidelity data from live markets:
- **Polymarket Scraper**: Multi-frequency ingestion (minutely/hourly/daily) for prediction market research.
- **WebSocket Hub**: Managed streaming for real-time price discovery and order flow.

---

## 🛠️ Development & Stewardship

The project maintains professional software standards across all languages:
- **Testing**: Comprehensive suites across Rust (Criterion benchmarks), Python (Pytest), and TypeScript.
- **Documentation**: 100% docstring/JSDoc coverage; verified with compliance checkers.
- **CI/CD**: Automated linting, type-checking, and build validation for every commit.

---

## 🚀 Future Roadmap

- **Multi-Agent Simulation**: Modeling competing RL agents in a single arena.
- **Distributed Training**: Scaling model learning across GPU clusters using Ray.
- **Microservices Shift**: Transitioning core components to specialized services for cloud scalability.
- **Causal Analysis**: Integrating causal inference to understand market drivers beyond simple correlation.

---

**NGLab** is built for the intersection of quantitative finance and cutting-edge AI research.
