# Contributing to NGLab

<a href="https://www.gnu.org/licenses/agpl-3.0"><img alt="License: AGPL v3" src="https://img.shields.io/badge/License-AGPL_v3-blue.svg"></a>
<a href="http://makeapullrequest.com"><img alt="PRs Welcome" src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square"></a>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
<a href="https://github.com/astral-sh/ruff"><img alt="Ruff" src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json"></a>

Thank you for your interest in contributing to NGLab! This document provides guidelines and instructions for contributing to the project.

> **Welcome, Engineer.**
> You are about to contribute to a high-frequency, multimodal intelligence platform. The standards here are high because the stakes are real. Use this document as your definitive guide to the **Process**, **Style**, and **Philosophy** of developing for NGLab.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Core Philosophy](#core-philosophy)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Development Workflow](#development-workflow)
- [Code Style](#code-style)
- [Testing](#testing)
- [RFC Process](#rfc-process)
- [Pull Request Process](#pull-request-process)
- [Release Process](#release-process)
- [Project Structure](#project-structure)

---

## Code of Conduct

Please be respectful and constructive in all interactions. We aim to foster an open and welcoming environment.

---

## Quick Start Checklist (First-Time Contributors)

> [!TIP]
> Complete this checklist to be ready for your first contribution!

- [ ] Fork and clone the repository
- [ ] Run `just setup` (installs all dependencies)
- [ ] Run `just test` (verify environment works)
- [ ] Run `just build-python` (build Rust extension)
- [ ] Browse issues labeled `good first issue`
- [ ] Read [DEVELOPMENT.md](../docs/DEVELOPMENT.md) for IDE setup
- [ ] Read [TESTING.md](../docs/TESTING.md) for testing guidelines
- [ ] Join the community Discord (link in README)

**Estimated time**: 30 minutes

---

## Core Philosophy

Before writing a single line of code, internalize these three axioms:

### 1. "Zero-Copy or Die"

Data movement is the enemy of latency. In the Rust-Python bridge, strictly prefer passing pointers over copying data.

- **BAD**: Serializing a Rust struct to JSON to read it in Python.
- **GOOD**: Exposing the raw memory address via `PyArray` and reading it with NumPy.

### 2. "Types are Documentation"

We do not write vague code.

- **Python**: Every function key must be typed. Use `mypy` strict mode.
- **Rust**: Use NewTypes (`struct Price(f64)`) to prevent unit confusion (e.g., preventing adding Price to Volume).

### 3. "Determinism is King"

The simulation must produce bit-exact results for the same seed, regardless of the hardware.

- Avoid iterating over `HashMap` (random order). Use `BTreeMap` or `IndexMap` instead.
- Avoid standard `RNG`. Use our seeded `ChaCha8` RNG wrapper.

---

## Getting Started

### Prerequisites

- **Rust**: 1.70+ (install via [rustup](https://rustup.rs/))
- **Python**: 3.11+ (recommended: use [UV](https://docs.astral.sh/uv/) or pyenv)
- **Node.js**: 20+ (for TypeScript/Tauri frontend)
- **Git**: For version control
- **Just**: Task runner (install via `cargo install just`)

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/nglab.git
   cd nglab
   ```
3. Add upstream remote:
   ```bash
   git remote add upstream https://github.com/ORIGINAL_OWNER/nglab.git
   ```

---

## Development Setup

### Automated Setup (Recommended)

Use the provided `just` command to set up everything:

```bash
just setup
```

This will:

- Install Rust toolchain components (rustfmt, clippy)
- Install Python dependencies
- Install TypeScript/Node.js dependencies
- Set up pre-commit hooks
- Install development tools

### Manual Setup

If you prefer manual setup or the automated script fails:

**Rust:**

```bash
rustup update stable
rustup component add rustfmt clippy
cargo build
```

**Python:**

```bash
cd python
uv sync  # or: pip install -e ".[dev]"
```

**TypeScript:**

```bash
cd typescript
npm ci
```

**Pre-commit Hooks:**

```bash
pip install pre-commit
pre-commit install
```

### VS Code Configuration

Copy this into your `.vscode/settings.json` for optimal DX:

```json
{
  "rust-analyzer.check.command": "clippy",
  "python.analysis.typeCheckingMode": "strict",
  "editor.formatOnSave": true,
  "[python]": {
    "editor.defaultFormatter": "charliermarsh.ruff"
  },
  "[rust]": {
    "editor.defaultFormatter": "rust-lang.rust-analyzer"
  }
}
```

---

## Development Workflow

### 1. Create a Branch

Create a feature branch from `main`:

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

Branch naming conventions:

- `feature/` - New features (e.g., `feature/mamba-backbone`)
- `fix/` - Bug fixes (e.g., `fix/clob-matching`)
- `docs/` - Documentation changes
- `refactor/` - Code refactoring
- `test/` - Test additions/changes
- `chore/` - Maintenance tasks
- `perf/` - Performance improvements

### 2. Make Changes

Follow the code style guidelines below. Run formatters before committing:

```bash
just fmt
```

### 3. Test Your Changes

Run all tests to ensure nothing is broken:

```bash
just test
```

Run specific test suites:

```bash
just test-rust
just test-python
just test-typescript
```

### 4. Lint Your Code

Ensure code passes linting:

```bash
just lint
```

Auto-fix linting issues:

```bash
just fix
```

### 5. Commit Your Changes

Write clear, descriptive commit messages following [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>

<body>

<footer>
```

Types:

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, no logic change)
- `refactor`: Code refactoring
- `test`: Test changes
- `chore`: Maintenance tasks

Examples:

```bash
git commit -m "feat(python): add VAE model for time series generation"
git commit -m "fix(rust): correct order matching logic in OrderBook"
git commit -m "docs: update README with installation instructions"
```

### 6. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub.

---

## Code Style

### Rust

- Follow [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)
- Use `rustfmt` for formatting (configured in `rustfmt.toml`)
- Use `clippy` for linting with no warnings
- Add documentation comments (`///`) for public APIs
- Write unit tests for new functionality

#### The "Do's and Don'ts"

| Context            | DO ✅                                               | DON'T ❌                                        |
| :----------------- | :-------------------------------------------------- | :---------------------------------------------- |
| **Error Handling** | Return `Result<T, AppError>`                        | Use `.unwrap()` or `.expect()` (Instant reject) |
| **Concurrency**    | Use `tokio::select!` for cancellation               | Use raw threads `std::thread::spawn`            |
| **Serialization**  | Use `serde` with `#[serde(rename_all="camelCase")]` | Manually format JSON strings                    |
| **Float Math**     | Use `f64` and handle `NaN` implicitly               | Use `f32` (precision loss in financial math)    |

**Example:**

```rust
/// Calculates the Sharpe ratio for a series of returns.
///
/// # Arguments
/// * `returns` - Slice of return values
/// * `risk_free_rate` - Risk-free rate (annualized)
///
/// # Returns
/// The Sharpe ratio
pub fn sharpe_ratio(returns: &[f64], risk_free_rate: f64) -> f64 {
    // Implementation
}
```

### Python

- Follow [PEP 8](https://peps.python.org/pep-0008/) and [PEP 257](https://peps.python.org/pep-0257/)
- Use `ruff` for formatting and linting (configured in `pyproject.toml`)
- Use type hints for all function signatures
- Add docstrings (Google style) for public APIs
- Maximum line length: 100 characters

#### The "Do's and Don'ts"

| Context     | DO ✅                                     | DON'T ❌                    |
| :---------- | :---------------------------------------- | :-------------------------- |
| **Typing**  | `def foo(x: float) -> list[int]:`         | `def foo(x):`               |
| **Config**  | Use `Hydra` for parameters                | Hardcode magic numbers      |
| **Loops**   | Use vectorized `numpy`/`torch` operations | Write `for` loops over data |
| **Imports** | `from typing import Annotated`            | `from typing import *`      |

**Example:**

```python
def calculate_sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
) -> float:
    """Calculate the Sharpe ratio for a series of returns.

    Args:
        returns: Array of return values.
        risk_free_rate: Risk-free rate (annualized).

    Returns:
        The Sharpe ratio.

    Raises:
        ValueError: If returns array is empty.
    """
    # Implementation
```

### TypeScript

- Follow [TypeScript guidelines](https://www.typescriptlang.org/docs/handbook/declaration-files/do-s-and-don-ts.html)
- Use `prettier` for formatting
- Use `eslint` for linting
- Add JSDoc comments for complex functions
- Use strict TypeScript configuration

#### The "Do's and Don'ts"

| Context    | DO ✅                               | DON'T ❌                          |
| :--------- | :---------------------------------- | :-------------------------------- |
| **State**  | Use `React Query` or `Tauri Events` | Use `useEffect` for data fetching |
| **Styles** | Use Tailwind utility classes        | Write raw CSS/SCSS files          |
| **Types**  | Define interfaces in `types/`       | Use `any` type (Instant reject)   |

**Example:**

```typescript
/**
 * Calculate the Sharpe ratio for a series of returns.
 * @param returns - Array of return values
 * @param riskFreeRate - Risk-free rate (annualized)
 * @returns The Sharpe ratio
 */
export function calculateSharpeRatio(
  returns: number[],
  riskFreeRate: number = 0.0,
): number {
  // Implementation
}
```

---

## Testing

### Test Coverage Requirements

- New features must include tests
- Bug fixes should include regression tests
- Aim for >70% code coverage
- Integration tests for Rust-Python interaction

### Running Tests

**All Tests:**

```bash
just test
```

**With Coverage:**

```bash
just coverage
```

**Specific Tests:**

```bash
# Rust
cargo test --test test_orderbook

# Python
cd python && pytest tests/test_vae.py -v

# TypeScript
cd typescript && npm test -- --testPathPattern=PriceChart
```

### Writing Tests

**Rust:**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_orderbook_creation() {
        let book = OrderBook::new();
        assert!(book.is_empty());
    }
}
```

**Python:**

```python
def test_vae_forward_pass():
    """Test VAE forward pass produces correct output shapes."""
    model = VAE(input_dim=10, latent_dim=5)
    x = torch.randn(32, 10)

    recon, mu, logvar = model(x)

    assert recon.shape == x.shape
    assert mu.shape == (32, 5)
    assert logvar.shape == (32, 5)
```

**TypeScript:**

```typescript
describe('PriceChart', () => {
  it('should render with initial data', () => {
    const { container } = render(<PriceChart data={mockData} />);
    expect(container.querySelector('canvas')).toBeInTheDocument();
  });
});
```

---

## RFC Process

For major architectural changes (e.g., "Switching from PPO to DreamerV3" or "Porting the GUI to Leptos"), you must submit an **Request For Comments (RFC)**.

1.  **Create an Issue**: Tag it `proposal/rfc`.
2.  **Draft the Document**: Create `rfc/000-my-proposal.md`.
3.  **Structure**:
    - **Summary**: 1-paragraph explanation.
    - **Motivation**: Why are we doing this?
    - **Design**: Technical implementation details.
    - **Drawbacks**: What specific problems does this introduce?
    - **Alternatives**: What else did you consider?

---

## Pull Request Process

### Before Submitting

1. **Sync with upstream:**

   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run pre-push checks:**

   ```bash
   just pre-push
   ```

   This runs: formatting, linting, and all tests.

3. **Update documentation** if needed (README, API docs, CLAUDE.md)

### PR Guidelines

- **Title**: Use conventional commit format: `feat: add VAE model`
- **Description**: Explain what and why, not how (code shows how)
- **Link Issues**: Reference related issues: `Closes #123`
- **Small PRs**: Keep changes focused and reviewable
- **Screenshots**: Include for UI changes
- **Breaking Changes**: Clearly document in PR description

### PR Template

```markdown
## Description

Brief description of changes

## Type of Change

- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing

How has this been tested?

## Checklist

- [ ] Code follows style guidelines
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] All tests passing
- [ ] No new warnings
```

### Review Process

- All PRs require at least one approval
- CI/CD must pass (Rust, Python, TypeScript checks)
- Address review comments promptly
- Maintain a single commit per logical change (squash if needed)

---

## Release Process

(Maintainers Only)

1.  **Bump Version**:
    - Update `Cargo.toml`
    - Update `package.json`
    - Update `pyproject.toml`
2.  **Changelog**: Add entry to `CHANGELOG.md`.
3.  **Tag**: `git tag -a v2.1.0 -m "Release v2.1.0"`
4.  **Push**: `git push origin --tags`
5.  Wait for CI to build and publish Docker images.

---

## Project Structure

```
nglab/
├── rust/               # Rust simulation engine
│   ├── src/
│   │   ├── simulation/ # TradingEnv, OrderBook, PolymarketArena
│   │   ├── models/     # Financial models (Black-Scholes, Heston, etc.)
│   │   ├── moon/       # Time series forecasting
│   │   └── web/        # Polymarket scraper
│   ├── Cargo.toml
│   └── tests/
├── python/             # Python training pipeline
│   ├── src/
│   │   ├── models/     # Deep learning models (VAE, GAN, Diffusion, etc.)
│   │   ├── pipeline/   # Training scripts and HPO
│   │   ├── agents/     # RL agents and wrappers
│   │   ├── policies/   # Trading strategies
│   │   └── env/        # Environment wrappers
│   ├── tests/
│   └── pyproject.toml
├── typescript/         # Tauri desktop app
│   ├── src/            # React frontend
│   │   ├── components/
│   │   ├── hooks/
│   │   └── App.tsx
│   ├── src-tauri/      # Rust backend for Tauri
│   └── package.json
├── .github/
│   ├── workflows/      # CI/CD pipelines
│   └── dependabot.yml
├── scripts/            # Utility scripts
├── justfile            # Task automation
└── IMPROVEMENT_PLAN.md # Roadmap
```

## Additional Resources

- **Architecture**: See [ARCHITECTURE.md](../docs/ARCHITECTURE.md) for system design
- **Tutorial**: See [TUTORIAL.md](../docs/TUTORIAL.md) for developer encyclopedia
- **Troubleshooting**: See [TROUBLESHOOTING.md](../docs/TROUBLESHOOTING.md) for common issues
- **API Docs**: Run `just docs` to generate documentation
- **Task Runner**: Run `just` to see all available commands

---

## Debugging Workflow

### 11.1 Debugging Rust Code

**Enable Debug Builds:**

```bash
# Build with debug symbols
cargo build

# Run with backtrace
RUST_BACKTRACE=1 cargo run --bin nglab-cli
```

**Using LLDB:**

```bash
# Start debugger
lldb target/debug/nglab-cli

# Set breakpoint
(lldb) b orderbook.rs:150
(lldb) run
```

**Logging:**

```rust
use tracing::{info, debug, warn, error};

// Add to your function
debug!(price = %order.price, "Processing order");
```

### 11.2 Debugging Python Code

**Using pdb:**

```python
import pdb; pdb.set_trace()  # Add breakpoint

# Or use the modern breakpoint() function
breakpoint()
```

**VS Code Launch Configuration:**

```json
{
  "name": "Python: Train PPO",
  "type": "debugpy",
  "request": "launch",
  "program": "${workspaceFolder}/python/src/pipeline/train_ppo.py",
  "console": "integratedTerminal",
  "env": {
    "PYTHONPATH": "${workspaceFolder}/python/src"
  }
}
```

### 11.3 Debugging TypeScript/Tauri

**Chrome DevTools:**

1. Run `npm run tauri dev`
2. Right-click in the app → Inspect
3. Use Console, Network, and Performance tabs

**Tauri Backend Logging:**

```rust
// In src-tauri/src/lib.rs
log::info!("Arena state: {:?}", state);
```

---

## Performance Optimization Guidelines

### 12.1 Rust Performance

| Technique            | When to Use            | Example                        |
| -------------------- | ---------------------- | ------------------------------ |
| **Pre-allocation**   | Known collection sizes | `Vec::with_capacity(1000)`     |
| **Iterators**        | Processing sequences   | Use `.iter()` over index loops |
| **SIMD**             | Numeric computations   | `packed_simd` or `std::simd`   |
| **Arena Allocation** | Many small objects     | `bumpalo` or `typed-arena`     |

**Profiling Tools:**

```bash
# Flamegraph
cargo install flamegraph
sudo cargo flamegraph --bin nglab-cli

# Criterion benchmarks
cargo bench --bench orderbook_bench
```

### 12.2 Python Performance

| Technique           | When to Use        | Example                         |
| ------------------- | ------------------ | ------------------------------- |
| **Vectorization**   | Array operations   | `np.dot(a, b)` over loops       |
| **JIT Compilation** | Hot functions      | `@torch.jit.script`             |
| **Async I/O**       | Network operations | `asyncio` / `aiohttp`           |
| **Data Loading**    | Training loops     | `num_workers > 0` in DataLoader |

**Profiling Tools:**

```python
# cProfile
python -m cProfile -o profile.prof train_ppo.py
snakeviz profile.prof

# PyTorch Profiler
with torch.profiler.profile() as prof:
    model(input)
print(prof.key_averages().table())
```

### 12.3 Memory Optimization

**Rust:**

- Use `Box<dyn Trait>` sparingly
- Prefer stack allocation for small structs
- Use `Cow<str>` for conditional ownership

**Python:**

- Use generators for large datasets
- Clear GPU memory: `torch.cuda.empty_cache()`
- Use `del` for large objects when done

---

## Security Best Practices

### 13.1 API Key Management

```bash
# NEVER do this
export API_KEY="sk-1234567890"  # In shell history!

# DO this instead
# Store in .env (git-ignored)
echo "API_KEY=sk-1234567890" >> .env

# Load in Python
from dotenv import load_dotenv
load_dotenv()
api_key = os.environ["API_KEY"]
```

### 13.2 Input Validation

**Rust:**

```rust
pub fn set_position(&mut self, position: f64) -> Result<(), ArenaError> {
    if position.is_nan() || position.is_infinite() {
        return Err(ArenaError::InvalidInput("Position must be finite"));
    }
    if position < self.config.min_position || position > self.config.max_position {
        return Err(ArenaError::InvalidInput("Position out of bounds"));
    }
    self.position = position;
    Ok(())
}
```

**Python:**

```python
def validate_config(config: dict) -> None:
    """Validate configuration before use."""
    required_keys = ["learning_rate", "batch_size", "num_envs"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")

    if config["learning_rate"] <= 0:
        raise ValueError("Learning rate must be positive")
```

### 13.3 Dependency Security

```bash
# Audit Rust dependencies
cargo audit

# Audit Python dependencies
pip-audit

# Audit npm dependencies
npm audit
```

---

## Architecture Decision Records (ADRs)

When making significant architectural decisions, document them using ADRs.

### ADR Template

Create files in `docs/adr/` with the following format:

```markdown
# ADR-001: Use Rust for Simulation Engine

## Status

Accepted

## Context

We need a simulation engine that can process >10,000 orders per second
with deterministic behavior across different platforms.

## Decision

We will use Rust for the simulation engine with PyO3 bindings for Python.

## Consequences

### Positive

- Microsecond-level latency
- Memory safety without GC
- Zero-copy data transfer to Python

### Negative

- Steeper learning curve for Python developers
- Longer compile times
- Need to maintain PyO3 bindings

## Alternatives Considered

1. **C++**: Rejected due to memory safety concerns
2. **Pure Python with Numba**: Rejected due to GIL limitations
3. **Go**: Rejected due to GC pauses

## References

- [PyO3 Documentation](https://pyo3.rs/)
- [Rust Performance Book](https://nnethercote.github.io/perf-book/)
```

### Existing ADRs

| ADR     | Title                   | Status   |
| ------- | ----------------------- | -------- |
| ADR-001 | Use Rust for Simulation | Accepted |
| ADR-002 | PyTorch over TensorFlow | Accepted |
| ADR-003 | Tauri over Electron     | Accepted |
| ADR-004 | Hydra for Configuration | Accepted |
| ADR-005 | TorchRL for RL          | Accepted |

---

## Mentorship & Onboarding

### For New Contributors

1. **Week 1**: Set up development environment, run all tests
2. **Week 2**: Pick a "good first issue" from GitHub
3. **Week 3**: Submit your first PR with mentor review
4. **Week 4**: Take on a more complex task

### Finding Mentors

- Check the `CODEOWNERS` file for area experts
- Join the Discord #mentorship channel
- Request a mentor in your first PR

---

## Recognition & Credits

Contributors are recognized in:

- `CHANGELOG.md` for each release
- `CONTRIBUTORS.md` (auto-generated from git history)
- GitHub release notes

---

## Questions?

If you have questions or need help:

1. Check existing issues and discussions
2. Open a new issue with the `question` label
3. Join our Discord community
4. Reach out to maintainers

**Response Time Expectations:**

- Issues: 24-48 hours for initial response
- PRs: 48-72 hours for first review
- Critical bugs: Same-day response

---

Thank you for contributing to NGLab! 🚀
