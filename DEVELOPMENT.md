# NGLab Development Guide

<a href="https://www.gnu.org/licenses/agpl-3.0"><img alt="License: AGPL v3" src="https://img.shields.io/badge/License-AGPL_v3-blue.svg"></a>
<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/Python-3.11-3776ab?logo=python&logoColor=white"></a>
<a href="https://www.rust-lang.org/"><img alt="Rust" src="https://img.shields.io/badge/Rust-1.80%2B-000000?logo=rust&logoColor=white"></a>
<a href="https://tauri.app/"><img alt="Tauri" src="https://img.shields.io/badge/Tauri-2.0-FFC131?logo=tauri&logoColor=white"></a>

> **Your complete reference for setting up, running, and debugging NGLab.**

## Table of Contents

1. [Quick Start](#quick-start)
2. [Environment Setup](#environment-setup)
3. [IDE Configuration](#ide-configuration)
4. [Environment Variables](#environment-variables)
5. [Local Development](#local-development)
6. [Building Components](#building-components)
7. [Database Setup](#database-setup)
8. [Performance Profiling](#performance-profiling)
9. [Cross-Language Debugging](#cross-language-debugging)
10. [Common Development Tasks](#common-development-tasks)

---

## Quick Start

```bash
# Clone and setup in 2 minutes
git clone https://github.com/acfharbinger/nglab.git
cd nglab
just setup          # Installs Rust, Python, TypeScript dependencies
just build-python   # Builds Rust extension for Python
just test           # Verifies everything works
```

---

## Environment Setup

### Prerequisites

| Tool        | Version | Installation                                                      |
| ----------- | ------- | ----------------------------------------------------------------- |
| **Rust**    | 1.75+   | `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \| sh` |
| **Python**  | 3.11+   | `uv python install 3.11` or system package manager                |
| **Node.js** | 20+     | `nvm install 20` or [nodejs.org](https://nodejs.org)              |
| **uv**      | Latest  | `curl -LsSf https://astral.sh/uv/install.sh \| sh`                |
| **Just**    | Latest  | `cargo install just`                                              |

### Automated Setup

The recommended way to set up your development environment:

```bash
just setup
```

This command:

1. Updates Rust toolchain and installs `rustfmt`, `clippy`
2. Creates Python virtualenv and installs dependencies via `uv sync`
3. Installs Node.js dependencies via `npm ci`
4. Sets up pre-commit hooks
5. Installs development tools (`cargo-cache`, `cargo-audit`)

### Manual Setup

If automated setup fails, follow these steps:

**Rust:**

```bash
rustup update stable
rustup component add rustfmt clippy
cargo build
```

**Python:**

```bash
cd python
uv sync                    # Creates .venv and installs deps
source ../.venv/bin/activate
maturin develop --release  # Builds Rust extension
```

**TypeScript:**

```bash
cd typescript
npm ci                     # Clean install
npm run build              # Build frontend
```

**Tauri Dependencies (Linux):**

```bash
# Ubuntu/Debian
sudo apt install libwebkit2gtk-4.1-dev libgtk-3-dev libappindicator3-dev

# Fedora
sudo dnf install webkit2gtk4.1-devel gtk3-devel libappindicator-gtk3-devel
```

---

## IDE Configuration

### VS Code (Recommended)

**Required Extensions:**

- `rust-analyzer` - Rust language support
- `Python` - Microsoft Python extension
- `ESLint` + `Prettier` - TypeScript linting/formatting
- `Tauri` - Tauri development tools
- `Even Better TOML` - Cargo.toml editing

**Workspace Settings (`.vscode/settings.json`):**

```json
{
  "rust-analyzer.cargo.features": "all",
  "rust-analyzer.checkOnSave.command": "clippy",
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
  "python.analysis.typeCheckingMode": "strict",
  "editor.formatOnSave": true,
  "[rust]": { "editor.defaultFormatter": "rust-lang.rust-analyzer" },
  "[python]": { "editor.defaultFormatter": "charliermarsh.ruff" },
  "[typescript]": { "editor.defaultFormatter": "esbenp.prettier-vscode" }
}
```

**Launch Configurations (`.vscode/launch.json`):**

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python: Train PPO",
      "type": "debugpy",
      "request": "launch",
      "program": "${workspaceFolder}/python/src/pipeline/train_ppo.py",
      "console": "integratedTerminal",
      "env": { "PYTHONPATH": "${workspaceFolder}/python/src" }
    },
    {
      "name": "Rust: Debug Tests",
      "type": "lldb",
      "request": "launch",
      "cargo": { "args": ["test", "--no-run"], "filter": { "kind": "test" } }
    },
    {
      "name": "Tauri: Dev",
      "type": "node",
      "request": "launch",
      "runtimeExecutable": "npm",
      "runtimeArgs": ["run", "tauri", "dev"],
      "cwd": "${workspaceFolder}/typescript"
    }
  ]
}
```

### PyCharm / RustRover

**PyCharm Setup:**

1. Open `python/` as project root
2. Set interpreter to `.venv/bin/python`
3. Mark `src/` as Sources Root
4. Enable pytest as test runner

**RustRover Setup:**

1. Open root directory containing `Cargo.toml`
2. Configure Clippy as external linter
3. Enable format on save with rustfmt

---

## Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

### Core Configuration

| Variable     | Default       | Description                                    |
| ------------ | ------------- | ---------------------------------------------- |
| `NGLAB_ENV`  | `development` | Environment: development, staging, production  |
| `DEBUG_MODE` | `false`       | Enable debug logging and assertions            |
| `LOG_LEVEL`  | `INFO`        | Logging verbosity: DEBUG, INFO, WARNING, ERROR |

### Database

| Variable      | Default     | Description                  |
| ------------- | ----------- | ---------------------------- |
| `DB_HOST`     | `localhost` | PostgreSQL host              |
| `DB_PORT`     | `5432`      | PostgreSQL port              |
| `DB_NAME`     | `nglab`     | Database name                |
| `DB_USER`     | `postgres`  | Database user                |
| `DB_PASSWORD` | -           | Database password (required) |

### GPU / Training

| Variable                | Default | Description                      |
| ----------------------- | ------- | -------------------------------- |
| `CUDA_VISIBLE_DEVICES`  | `0`     | GPU device IDs (comma-separated) |
| `NGLAB_DEVICE`          | `cuda`  | Training device: cuda, cpu, mps  |
| `NGLAB_BATCH_SIZE`      | `64`    | Training batch size              |
| `NGLAB_MIXED_PRECISION` | `true`  | Enable FP16/BF16 training        |

### Monitoring

| Variable                      | Default                 | Description              |
| ----------------------------- | ----------------------- | ------------------------ |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://localhost:4317` | Jaeger/OTLP endpoint     |
| `WANDB_API_KEY`               | -                       | Weights & Biases API key |
| `MLFLOW_TRACKING_URI`         | `http://localhost:5000` | MLflow server URI        |

---

## Local Development

### Development Servers

**Tauri Desktop App (Hot Reload):**

```bash
just dev
# or: cd typescript && npm run tauri dev
```

**Python API Server:**

```bash
cd python
uvicorn src.api.app:app --reload --port 8000
```

**Jupyter Notebooks:**

```bash
just notebook
# or: cd python && jupyter notebook
```

### Watch Mode

**Rust (auto-rebuild on change):**

```bash
just watch
# or: cargo watch -x check -x test
```

**TypeScript (Vite hot reload):**

```bash
cd typescript && npm run dev
```

### Code Quality Loop

```bash
# Format → Lint → Test (recommended before commits)
just pre-push

# Or individually:
just fmt      # Format all code
just lint     # Check linting
just test     # Run all tests
just fix      # Auto-fix linting issues
```

---

## Building Components

### Build Everything

```bash
just build  # Builds Rust, Python extension, TypeScript
```

### Individual Components

| Component            | Command                 | Output                       |
| -------------------- | ----------------------- | ---------------------------- |
| **Rust Library**     | `just build-rust`       | `target/release/libnglab.so` |
| **Python Extension** | `just build-python`     | `python/nglab/*.so`          |
| **TypeScript/Web**   | `just build-typescript` | `typescript/dist/`           |
| **Tauri Desktop**    | `just build-tauri`      | Platform-specific binary     |

### Release Builds

```bash
# Production Rust build
cargo build --release --all-features

# Production Python package
cd python && maturin build --release

# Production Tauri app
cd typescript && npm run tauri build
```

---

## Database Setup

### Local PostgreSQL

**Docker (Recommended):**

```bash
docker run -d \
  --name nglab-postgres \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=password \
  -e POSTGRES_DB=nglab \
  -p 5432:5432 \
  postgres:16
```

**System Install:**

```bash
# Ubuntu
sudo apt install postgresql
sudo -u postgres createdb nglab

# macOS
brew install postgresql@16
createdb nglab
```

### Redis Cache

```bash
docker run -d --name nglab-redis -p 6379:6379 redis:7-alpine
```

### Full Local Stack

```bash
docker compose up -d  # Starts PostgreSQL, Redis, Jaeger
```

### Database Migrations

```bash
# Apply migrations
cd migrations && psql -U postgres -d nglab -f V001__initial.sql

# Reset database
just reset-credentials
```

---

## Performance Profiling

### Rust Profiling

**Flamegraph (CPU):**

```bash
just profile orderbook
# or: cargo flamegraph --bench orderbook
```

**Criterion Benchmarks:**

```bash
just bench
# or: cargo bench
```

**Memory (Valgrind):**

```bash
valgrind --tool=massif target/release/nglab
ms_print massif.out.*
```

### Python Profiling

**py-spy (CPU sampling):**

```bash
just profile-python python/src/pipeline/train_ppo.py
# or: py-spy record -o profile.svg -- python script.py
```

**cProfile (deterministic):**

```bash
python -m cProfile -o profile.prof python/src/pipeline/train_ppo.py
snakeviz profile.prof
```

**Memory Profiling:**

```bash
python -m memory_profiler python/src/main.py
```

**PyTorch Profiler:**

```python
import torch.profiler as profiler

with profiler.profile(activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA]) as prof:
    model(input)

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

### GPU Monitoring

```bash
# Real-time GPU usage
watch -n 1 nvidia-smi

# PyTorch memory summary
python -c "import torch; print(torch.cuda.memory_summary())"
```

---

## Cross-Language Debugging

### Rust ↔ Python (PyO3)

**Debug Rust from Python:**

```bash
# Build with debug symbols
cd python && maturin develop

# Run Python with Rust debugger attached
rust-gdb --args python your_script.py
```

**Common Issues:**

- `ImportError: libnglab.so` → Run `maturin develop`
- Segfault in Rust → Enable `RUST_BACKTRACE=full`

### Rust ↔ TypeScript (Tauri)

**Debug Tauri Backend:**

```bash
cd typescript
RUST_BACKTRACE=1 npm run tauri dev
```

**Inspect IPC Events:**

```typescript
// In React component
import { listen } from "@tauri-apps/api/event";

useEffect(() => {
  const unlisten = listen("arena-update", (event) => {
    console.log("Received:", event.payload);
  });
  return () => {
    unlisten.then((f) => f());
  };
}, []);
```

---

## Common Development Tasks

### Adding a New Python Module

```bash
# 1. Create module
mkdir -p python/src/mymodule
touch python/src/mymodule/__init__.py

# 2. Add to pyproject.toml if needed

# 3. Run tests
cd python && pytest tests/unit/test_mymodule.py -v
```

### Adding a New Rust Module

```bash
# 1. Create module file
touch rust/src/mymodule.rs

# 2. Register in lib.rs
echo "pub mod mymodule;" >> rust/src/lib.rs

# 3. Expose to Python via PyO3 (if needed)
# Add #[pymodule] function in lib.rs

# 4. Rebuild Python extension
just build-python
```

### Adding a React Component

```bash
# 1. Create component
touch typescript/src/components/MyComponent.tsx

# 2. Add to App.tsx or relevant parent

# 3. Test
cd typescript && npm test
```

### Running Specific Tests

```bash
# Rust - single test
cargo test test_orderbook_matching

# Python - single file
cd python && pytest tests/unit/test_vae.py -v

# Python - single test
cd python && pytest tests/unit/test_vae.py::test_forward_pass -v

# TypeScript
cd typescript && npm test -- --grep "MyComponent"
```

---

## Troubleshooting Development Issues

| Issue                               | Solution                                                |
| ----------------------------------- | ------------------------------------------------------- |
| `ModuleNotFoundError: nglab`        | Run `just build-python`                                 |
| Rust build fails with linker errors | Install system deps: `sudo apt install build-essential` |
| TypeScript types out of sync        | Run `npm run build` to regenerate                       |
| Pre-commit hooks failing            | Run `just fix` to auto-fix                              |
| GPU not detected                    | Check `CUDA_VISIBLE_DEVICES` and `nvidia-smi`           |

For more issues, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

---

**Happy Coding! 🚀**
