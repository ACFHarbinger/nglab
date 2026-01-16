# Contributing to NGLab

Thank you for your interest in contributing to NGLab! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Development Workflow](#development-workflow)
- [Code Style](#code-style)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Project Structure](#project-structure)

## Code of Conduct

Please be respectful and constructive in all interactions. We aim to foster an open and welcoming environment.

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

## Development Workflow

### 1. Create a Branch

Create a feature branch from `main`:

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

Branch naming conventions:
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation changes
- `refactor/` - Code refactoring
- `test/` - Test additions/changes
- `chore/` - Maintenance tasks

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

## Code Style

### Rust

- Follow [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)
- Use `rustfmt` for formatting (configured in `rustfmt.toml`)
- Use `clippy` for linting with no warnings
- Add documentation comments (`///`) for public APIs
- Write unit tests for new functionality

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
  riskFreeRate: number = 0.0
): number {
  // Implementation
}
```

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

- **Architecture**: See [CLAUDE.md](CLAUDE.md) for tech stack overview
- **Improvement Plan**: See [IMPROVEMENT_PLAN.md](IMPROVEMENT_PLAN.md) for roadmap
- **API Docs**: Run `just docs` to generate documentation
- **Task Runner**: Run `just` to see all available commands

## Questions?

If you have questions or need help:
1. Check existing issues and discussions
2. Open a new issue with the `question` label
3. Reach out to maintainers

---

Thank you for contributing to NGLab! 🚀
