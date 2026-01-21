<div align="center">

<img src="https://raw.githubusercontent.com/acfharbinger/nglab/main/assets/images/logo-nglab.png" alt="NGLAB Logo" style="width: 35%; height: auto;">

# Nothing Gambles Like A Bot (NGLAB)

**A Comprehensive Multimodal Intelligence Framework for Quantitative Finance and Automated Gambling.**

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://github.com/pytorch/rl"><img alt="base: TorchRL" src="https://img.shields.io/badge/base-TorchRL-red"></a>
<a href="https://hydra.cc/"><img alt="config: Hydra" src="https://img.shields.io/badge/config-Hydra-89b8cd"></a>
<a href="https://github.com/astral-sh/ruff"><img alt="Code style: ruff" src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json"></a>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
<a href="https://colab.research.google.com/"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
<a href="https://pypi.org/"><img alt="PyPI" src="https://img.shields.io/pypi/v/rl4co?logo=pypi"></a>
<a href="https://app.codecov.io/github/acfharbinger/nglab/tree/main/python">
  <img alt="Python Coverage" src="https://img.shields.io/codecov/c/github/acfharbinger/nglab?flag=python&logo=python&label=python%20cov">
</a>
<a href="https://app.codecov.io/github/acfharbinger/nglab/tree/main/rust">
  <img alt="Rust Coverage" src="https://img.shields.io/codecov/c/github/acfharbinger/nglab?flag=rust&logo=rust&label=rust%20cov&logoColor=white">
</a>
<a href="https://app.codecov.io/github/acfharbinger/nglab/tree/main/typescript">
  <img alt="TypeScript Coverage" src="https://img.shields.io/codecov/c/github/acfharbinger/nglab?flag=typescript&logo=typescript&label=ts%20cov">
</a>
<a href="https://github.com/acfharbinger/nglab/actions/workflows/ci.yml"><img alt="Test" src="https://github.com/acfharbinger/nglab/actions/workflows/ci.yml/badge.svg"></a>

</br>

<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/Python-3.11-3776ab?logo=python&logoColor=white"></a>
<a href="https://www.typescriptlang.org/"><img alt="TypeScript" src="https://img.shields.io/badge/TypeScript-3178C6?logo=typescript&logoColor=white"></a>
<a href="https://react.dev/"><img alt="React" src="https://img.shields.io/badge/React-61DAFB?logo=react&logoColor=black"></a>
<a href="https://www.rust-lang.org/"><img alt="Rust" src="https://img.shields.io/badge/Rust-000000?logo=rust&logoColor=white"></a>
<a href="https://tauri.app/"><img alt="Tauri" src="https://img.shields.io/badge/Tauri-FFC131?logo=tauri&logoColor=white"></a>
<a href="https://github.com/astral-sh/uv"><img alt="uv" src="https://img.shields.io/badge/managed%20by-uv-261230.svg"></a>
<a href="https://www.docker.com/"><img alt="Docker" src="https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white"></a>

<p>

<p>
  <a href="#documentation-hub"><strong>📚 Documentation</strong></a> |
  <a href="#overview"><strong>Overview</strong></a> |
  <a href="#key-features"><strong>Features</strong></a> |
  <a href="#quickstart"><strong>Quickstart</strong></a> |
  <a href="#learning-paradigms"><strong>Paradigms</strong></a> |
  <a href="#model-ecosystem"><strong>Models</strong></a> |
  <a href="#datasets"><strong>Datasets</strong></a> |
  <a href="#setup-dependencies"><strong>Setup</strong></a> |
  <a href="#execute-program"><strong>Usage</strong></a>
</p>

</div>

---

## 📚 Documentation Hub

Start here! We have expanded our documentation to cover every aspect of the system.

| Document | Description | Target Audience |
| :--- | :--- | :--- |
| **[TUTORIAL.md](TUTORIAL.md)** | **The Developer Encyclopedia.** Exhaustive deep dives into every module, code snippets, and implementation details. | Developers, Contributors |
| **[AGENTS.md](AGENTS.md)** | **The Strategy Handbook.** Classification of Agents (RL vs Heuristic), Policy Architectures (Mamba/CNN), and Observation Spaces. | Quants, Researchers |
| **[ARCHITECTURE.md](ARCHITECTURE.md)** | **The System Blueprint.** High-level design, data flow diagrams, system boundaries, and deployment topology. | Architects, DevOps |
| **[CONTRIBUTING.md](CONTRIBUTING.md)** | **The Developer Handbook.** Code style, PR process, RFC workflow, and release procedures. | Contributors |
| **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** | **The Field Repair Manual.** Common issues, diagnostic steps, and quick fixes. | Everyone |
| **[IMPROVEMENTS.md](IMPROVEMENTS.md)** | **The Roadmap.** Active tasks, feature requests, and the long-term vision for the platform. | Project Managers |

---

## Overview

**NGLAB** is a high-performance framework designed to navigate the complexities of financial markets. It combines state-of-the-art **Deep Learning**, **Classical Machine Learning**, and **Reinforcement Learning** to fuse numerical price data with global sentiment analysis.

NGLAB isn't just a trading bot; it's an end-to-end research and execution arena that decouples financial engineering from model experimentation. The platform provides:

- **High-fidelity market simulation** with microsecond-level order book dynamics
- **Modular ML pipeline** supporting 30+ model architectures
- **Real-time visualization** through a modern desktop application
- **Production-ready infrastructure** with Docker, Kubernetes, and CI/CD support

---

## Key Features

### 🚀 Performance

| Metric | Achievement |
|--------|-------------|
| **Order Matching** | < 100μs latency |
| **Environment Steps** | > 20,000 steps/second |
| **Data Bridge** | Zero-copy NumPy via PyO3 |
| **UI Rendering** | Locked 60 FPS |

### 🧠 Intelligence

- **30+ Model Architectures**: From classical ARIMA to cutting-edge Mamba (SSM)
- **Multiple RL Algorithms**: PPO, SAC, DQN with TorchRL integration
- **Automated HPO**: DEHB (Differential Evolution Hyperband) for efficient hyperparameter search
- **Meta-Learning**: MAML for rapid adaptation to new market regimes

### 🎯 Simulation

- **Central Limit Order Book (CLOB)**: Full price-time priority matching
- **Advanced Order Types**: Iceberg, Trailing Stop, Stop-Loss, Take-Profit
- **Multi-Asset Support**: Trade portfolios across multiple instruments
- **Polymarket Integration**: Prediction market simulation and data scraping

### 🖥️ User Experience

- **Cross-Platform Desktop App**: Built with Tauri 2.0 (Rust + React)
- **Real-Time Charts**: High-performance visualization with lightweight-charts
- **Interactive Analysis**: EDA tools, backtesting, and model comparison

---

## Quickstart

Get up and running in 5 minutes:

```bash
# 1. Clone the repository
git clone https://github.com/acfharbinger/nglab.git
cd nglab

# 2. Run automated setup
just setup

# 3. Build the Rust extension
just build-python

# 4. Start training
python python/src/pipeline/train_ppo.py

# 5. Launch the GUI (optional)
cd typescript && npm run tauri dev
```

### Verify Installation

```bash
# Check all components
just check-health

# Run tests
just test
```

---

## Learning Paradigms

NGLAB implements a modular pipeline system supporting diverse learning strategies:

| Paradigm | Description | Use Case |
|----------|-------------|----------|
| **Reinforcement Learning** | Vectorized environments for portfolio optimization | Trading agent training |
| **Supervised Learning** | High-precision forecasting and regression | Price prediction |
| **Self-Supervised** | Latent representation learning using VAEs | Market regime detection |
| **Unsupervised** | Clustering and anomaly detection | Outlier identification |
| **Semi-Supervised** | Leveraging unlabeled data with sparse labels | Label-efficient learning |
| **Meta-Learning** | Rapid adaptation via MAML | Market regime shifts |
| **Online Learning** | Continuous adaptation to market drift | Live trading |

---

## Model Ecosystem

We provide a vast library of models categorized into two primary families:

### 🏛️ Classical Machine Learning (Mac Models)

Efficient, interpretable, and robust baseline models:

| Category | Models |
|----------|--------|
| **Trees & Ensembles** | Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost |
| **Linear & Kernel** | Ridge, Lasso, ElasticNet, SVM |
| **Probabilistic** | Naive Bayes, Bayesian Networks |
| **Instance-based** | K-Nearest Neighbors (KNN) |
| **Time Series** | ARIMA, GARCH, Exponential Smoothing, Prophet |

### 🧠 Deep Learning (Deep Models)

Modern neural architectures specialized for sequence and multimodal data:

| Category | Models |
|----------|--------|
| **Attention** | NS-Transformers, Multi-Head Attention |
| **State Space** | Mamba (SSM), S4, Liquid Neural Networks |
| **Recurrent** | LSTM, GRU, xLSTM |
| **Generative** | VAE, TimeGAN, Diffusion U-Net, Flow-based |
| **Advanced** | Neural ODE, PINN, DNC, SNN |

---

## Datasets

NGLAB leverages a diverse range of data sources for training and inference:

### 📈 Stock Market Data

| Dataset | Description | Source |
|---------|-------------|--------|
| Stock Market Dataset | Stocks and ETFs from 1999 to 2020 | [Kaggle](https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset) |
| S&P 500 | Historical index data | Yahoo Finance |
| Crypto | Bitcoin, Ethereum tick data | Binance API |

### 📰 Textual & Sentiment Data

| Dataset | Description | Source |
|---------|-------------|--------|
| BBC News | News articles 2004-2005 | [UCD](http://mlg.ucd.ie/datasets/bbc.html) |
| Australia News | News from 2003-2021 | [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/SYBGZL) |
| World Politics | International news | [NewsData.io](https://newsdata.io/files/datasets/world-politics-news) |
| Global News | Multi-source news | [Kaggle](https://www.kaggle.com/datasets/everydaycodings/global-news-dataset) |
| Twitter/X | Social media sentiment | [Kaggle](https://www.kaggle.com/datasets/bhavikjikadara/tweets-dataset/data) |

### 🎲 Prediction Markets

| Dataset | Description | Source |
|---------|-------------|--------|
| Polymarket | Real-time prediction market data | Polymarket API |
| Metaculus | Historical forecasting data | Metaculus |

---

## Project Structure

```
nglab/
├── rust/                 # 🦀 Rust simulation engine
│   ├── src/
│   │   ├── simulation/   # TradingEnv, OrderBook, PolymarketArena
│   │   ├── models/       # Black-Scholes, Heston, Credit Risk
│   │   ├── moon/         # ARIMA, GARCH, Prophet
│   │   └── web/          # Polymarket scraper
│   └── Cargo.toml
├── python/               # 🐍 Python ML pipeline
│   ├── src/
│   │   ├── models/       # VAE, GAN, Diffusion, Mamba, etc.
│   │   ├── pipeline/     # Training scripts, HPO
│   │   ├── agents/       # RL agents, wrappers
│   │   ├── policies/     # Trading strategies
│   │   └── env/          # Environment wrappers
│   └── pyproject.toml
├── typescript/           # 📱 Tauri desktop app
│   ├── src/              # React frontend
│   │   ├── components/   # UI components
│   │   ├── hooks/        # React hooks
│   │   └── App.tsx
│   └── src-tauri/        # Rust backend for Tauri
├── deploy/               # 🚀 Deployment configs
│   ├── docker/           # Docker configurations
│   ├── k8s/              # Kubernetes manifests
│   └── helm/             # Helm charts
├── scripts/              # 🛠️ Utility scripts
└── .github/              # CI/CD workflows
```

---

## Setup Dependencies

You can choose to install this repository's dependencies using any of the following methods:

### ⚡ UV (Recommended)

Fastest setup using the [uv package manager](https://github.com/astral-sh/uv).

```bash
# Sync the project and create a virtual environment
uv sync

# Activate the environment
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate.bat # Windows CMD
```

To deactivate and/or delete the created virtual environment:
```bash
deactivate
rm -rf .venv
```

### 🐍 Anaconda

```bash
conda env create --file env/environment.yml -y --name wsr
conda activate wsr
```

### 📦 Virtual Environment (Standard)

> [!NOTE]
> This method requires Python 3.11 pre-installed on your system.

```bash
python3 -m venv env/.wsr
source env/.wsr/bin/activate
pip install -r env/requirements.txt
pip install -r env/pip_requirements.txt
```

To deactivate and/or delete the created virtual environment:
```bash
deactivate
rm -rf env/.wsr
```

### 🐳 Docker

```bash
# Build the image
docker build -f Dockerfile.prod -t nglab:latest .

# Run the container
docker run --gpus all -p 8080:8080 nglab:latest
```

---

## Execute Program

Choose your preferred interface to interact with the program!

### 🛠️ Terminal User Interface (TUI)

Run the Rust-based CLI for high-performance operations:
```bash
cargo run --bin nglab-cli
```

### 🖥️ Graphical User Interface (GUI)

Run the TypeScript/Tauri GUI for a modern, cross-platform experience:
```bash
cd typescript
npm run tauri dev
```

### 🐍 Python Scripts

Run training and inference directly:
```bash
# Train a PPO agent
python python/src/pipeline/train_ppo.py

# Run inference
python python/src/pipeline/infer.py --model checkpoint.pt

# Hyperparameter optimization
python python/src/pipeline/hpo/run_dehb.py
```

### 📊 Just Commands

Use the task runner for common operations:
```bash
just              # List all available commands
just setup        # Full environment setup
just build        # Build all components
just test         # Run all tests
just lint         # Run linters
just fmt          # Format code
just dev          # Start development environment
just clean        # Clean build artifacts
```

---

## Setup Scripts

For a fully automated environment setup, use the provided scripts:

```bash
# Environment setup (choose method)
bash scripts/setup_env.sh <uv|conda|venv>

# Build Rust components
bash scripts/build_rust.sh

# Build Docker images
bash scripts/build_docker.sh

# Database backup
bash scripts/backup_db.sh

# Cleanup
bash scripts/cleanup.sh
```

---

## Performance Benchmarks

| Component | Metric | Target | Achieved |
|-----------|--------|--------|----------|
| OrderBook Insert | Latency | < 1ms | ~0.1ms |
| TradingEnv Step | Latency | < 1ms | ~0.5ms |
| Order Matching | Throughput | > 10k ops/sec | ~50k ops/sec |
| Memory Usage | RAM | < 100MB | ~50MB |
| Model Forward | Latency | < 10ms | ~5ms (Mamba) |
| Training Step | Latency | < 100ms | ~80ms |

---

## Contributing

We welcome contributions! Please read our [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:

- Code style and linting
- Pull request process
- RFC workflow for major changes
- Release procedures

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [PyTorch](https://pytorch.org/) - Deep learning framework
- [TorchRL](https://github.com/pytorch/rl) - Reinforcement learning library
- [Tauri](https://tauri.app/) - Desktop application framework
- [Hydra](https://hydra.cc/) - Configuration management
- [Mamba](https://github.com/state-spaces/mamba) - State space model architecture

---

<div align="center">
<strong>NGLab</strong> - Where AI meets the markets
</div>