# NGLab Dependencies

<a href="https://www.gnu.org/licenses/agpl-3.0"><img alt="License: AGPL v3" src="https://img.shields.io/badge/License-AGPL_v3-blue.svg"></a>

This document provides a comprehensive list of the core dependencies used across the NGLab platform.

## System Requirements

| Component   | Version | Purpose                                    |
| :---------- | :------ | :----------------------------------------- |
| **Rust**    | 1.80+   | Core simulation engine and PyO3 extensions |
| **Python**  | 3.11+   | Machine learning and orchestration         |
| **Node.js** | 20+     | Frontend development                       |
| **Docker**  | 24+     | Containerization and deployment            |
| **CUDA**    | 11.8+   | GPU acceleration (optional)                |

---

## Backend (Rust)

**File**: `rust/Cargo.toml`

### Core Runtime

- **tokio** (`1.42`): Asynchronous runtime
- **futures-util** (`0.3`): Async utilities
- **pyo3** (`0.27`): Python bindings
- **numpy** (`0.27`): NumPy integration

### Data & Serialization

- **serde** (`1.0`): Serialization framework
- **serde_json** (`1.0`): JSON support
- **ndarray** (`0.17`): N-dimensional arrays
- **csv** (`1.3`): CSV processing
- **rusqlite** (`0.32`): SQLite database

### Utilities

- **tracing** (`0.1`): Instrumentation
- **config** (`0.14`): Configuration loader
- **rand** (`0.9`): Random number generation
- **chrono** (`0.4`): Date and time
- **sysinfo** (`0.33`): System monitoring

### TUI & Visualization

- **ratatui** (`0.30`): Terminal UI
- **rerun** (`0.28.2`): Multimodal visualization

---

## Intelligence (Python)

**File**: `pyproject.toml`

### Deep Learning

- **torch** (`>=2.3`): Core DL framework
- **torchrl** (`>=0.10.1`): Reinforcement learning
- **pytorch-lightning** (`==2.0`): Training framework
- **torch-geometric** (`>=2.7.0`): Graph neural networks

### Data Science

- **numpy** (`==1.24`): Numerical computing
- **pandas** (`==2.0`): Data manipulation
- **scikit-learn** (`>=1.3.0`): Classical ML
- **scipy** (`==1.10`): Scientific computing

### Infrastructure

- **hydra-core** (`>=1.3.2`): Configuration management
- **optuna** (`>=4.6.0`): Hyperparameter optimization
- **wandb** (`==0.15`): Experiment tracking
- **ray** (`>=2.7.0`): Distributed computing

---

## Frontend (TypeScript)

**File**: `typescript/package.json`

### Core Frameworks

- **react** (`^19.1.0`): UI library
- **vite** (`^7.0.4`): Build tool
- **tauri** (`^2.0`): Desktop application framework

### UI Components

- **tailwindcss** (`^4.1.18`): Utility-first CSS
- **lucide-react** (`^0.562.0`): Icons
- **clsx** (`^2.1.1`): Class name utility

### Visualization

- **midweight-charts** (`^5.1.0`): Financial charting
- **highcharts** (`^12.4.0`): Interactive charts
- **recharts** (`^3.7.0`): Composable charting library

---

## Development Tools

| Tool           | Purpose                                    |
| :------------- | :----------------------------------------- |
| **uv**         | Fast Python package installer and resolver |
| **just**       | Task runner (Makefile alternative)         |
| **pre-commit** | Git hooks management                       |
| **maturin**    | Rust/Python build tool                     |
| **vitest**     | Frontend testing framework                 |
