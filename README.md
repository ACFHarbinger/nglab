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
<a href="https://app.codecov.io/github/acfharbinger/nglab"><img alt="Codecov" src="https://codecov.io/github/acfharbinger/nglab/branch/main/badge.svg"></a>
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
  <a href="#overview"><strong>Overview</strong></a> |
  <a href="#learning-paradigms"><strong>Paradigms</strong></a> |
  <a href="#model-ecosystem"><strong>Model Ecosystem</strong></a> |
  <a href="#setup-dependencies"><strong>Setup</strong></a> |
  <a href="#execute-program"><strong>Usage</strong></a>
</p>

</div>

---

# Overview

**NGLAB** is a high-performance framework designed to navigate the complexities of financial markets. It combines state-of-the-art **Deep Learning**, **Classical Machine Learning**, and **Reinforcement Learning** to fuse numerical price data with global sentiment analysis.

NGLAB isn't just a trading bot; it's an end-to-end research and execution arena that decouples financial engineering from model experimentation.

---

## Learning Paradigms

NGLAB implements a modular pipeline system supporting diverse learning strategies:

* **Reinforcement Learning**: Vectorized environments for portfolio optimization and decision-making.
* **Supervised Learning**: High-precision forecasting and regression for price action.
* **Self-Supervised & Unsupervised**: Latent representation learning using VAEs, GANs, and clustering to detect market regimes.
* **Semi-Supervised Learning**: Leveraging vast amounts of unlabeled market data with sparse labels.
* **Meta-Learning**: Agents that adapt rapidly to new market conditions via MAML (Model-Agnostic Meta-Learning).
* **Online Learning**: Continuous adaptation to market drift in real-time.

---

## Model Ecosystem

We provide a vast library of models categorized into two primary families:

### 🏛️ Classical Machine Learning (Mac Models)
Efficient, interpretable, and robust baseline models:
* **Trees & Ensembles**: Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost.
* **Linear & Kernel**: Ridge, Lasso, ElasticNet, and Support Vector Machines (SVM).
* **Probabilistic**: Naive Bayes and Bayesian Networks.
* **Instance-based**: K-Nearest Neighbors (KNN).

### 🧠 Deep Learning (Deep Models)
Modern neural architectures specialized for sequence and multimodal data:
* **Attention & Transformers**: NS-Transformers and custom multi-head attention blocks.
* **State Space Models**: Implementation of **Mamba** blocks for long-range time-series dependencies.
* **Recurrent & Memory**: xLSTM, GRU, and Neural Turing Machines.
* **Probabilistic & Generative**: Diffusion U-Nets, Flow-based models, and Variational Autoencoders.
* **Bio-inspired**: Spiking Neural Networks (SNN) for low-latency signal processing.

---

## Datasets

NGLAB leverages a diverse range of data sources for training and inference:

### 📈 Stock Market Data
* [Stocks and ETFs from 1999 to 2020](https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset)

### 📰 Textual & Sentiment Data
* **News Articles**: [BBC News (2004-2005)](http://mlg.ucd.ie/datasets/bbc.html), [Australia News (2003-2021)](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/SYBGZL).
* **Global Context**: [World Politics News](https://newsdata.io/files/datasets/world-politics-news), [Global News Dataset](https://www.kaggle.com/datasets/everydaycodings/global-news-dataset).
* **Social Media**: [Twitter/X Tweets Dataset](https://www.kaggle.com/datasets/bhavikjikadara/tweets-dataset/data).

---

## Setup Dependencies

You can choose to install this repository's dependencies using any of the following methods:

### ⚡ UV (Recommended)
Fastest setup using the [uv package manager](https://github.com/astral-sh/uv).

```bash
# Sync the project and create a virtual environment
uv sync

# Activate the environment
source .venv/bin/activate  # Linux
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
[!NOTE] This method requires Python 3.11 pre-installed on your system.
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
npm run tauri dev
```

## Setup Scripts
For a fully automated environment setup, use the provided scripts:
```bash
bash scripts/setup_env.sh <uv|conda|venv>
bash build_rust.sh
bash build_docker.sh
bash backup_db.sh
bash cleanup.sh
```