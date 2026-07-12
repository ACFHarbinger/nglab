# NGLab Testing Guide

<a href="https://www.gnu.org/licenses/agpl-3.0"><img alt="License: AGPL v3" src="https://img.shields.io/badge/License-AGPL_v3-blue.svg"></a>
<a href="https://github.com/acfharbinger/nglab/actions/workflows/ci.yml"><img alt="Test" src="https://github.com/acfharbinger/nglab/actions/workflows/ci.yml/badge.svg"></a>
<a href="https://app.codecov.io/github/acfharbinger/nglab"><img alt="codecov" src="https://codecov.io/gh/acfharbinger/nglab/branch/main/graph/badge.svg"></a>
<a href="https://pytest.org/"><img alt="Pytest" src="https://img.shields.io/badge/Pytest-0A9EDC?logo=pytest&logoColor=white"></a>
<a href="https://vitest.dev/"><img alt="Vitest" src="https://img.shields.io/badge/Vitest-6E9F18?logo=vitest&logoColor=white"></a>

> **Quality is not negotiable.** This document defines our testing philosophy, strategies, and best practices.

## Table of Contents

1. [Testing Philosophy](#testing-philosophy)
2. [Test Organization](#test-organization)
3. [Running Tests](#running-tests)
4. [Coverage Requirements](#coverage-requirements)
5. [Rust Testing](#rust-testing)
6. [Python Testing](#python-testing)
7. [TypeScript Testing](#typescript-testing)
8. [Integration Testing](#integration-testing)
9. [Performance Benchmarks](#performance-benchmarks)
10. [Mocking Strategies](#mocking-strategies)
11. [CI/CD Integration](#cicd-integration)
12. [Test Data Management](#test-data-management)

---

## Testing Philosophy

### Core Principles

1. **Tests are Documentation**
   - A well-written test explains what the code should do
   - Prefer descriptive test names over comments

2. **Fast Feedback Loops**
   - Unit tests complete in < 1 second
   - Full test suite completes in < 5 minutes
   - Flaky tests are immediately fixed or deleted

3. **Property-Based Testing**
   - Use Hypothesis (Python) and proptest (Rust) for edge case discovery
   - Generate random inputs to find unexpected failures

4. **Test at the Right Level**
   - Unit tests: Fast, isolated, abundant
   - Integration tests: Moderate speed, test component boundaries
   - E2E tests: Slow, few, cover critical paths

### The Testing Pyramid

```
       /\
      /  \      E2E Tests (5%)
     /----\     Tauri app, full pipelines
    /      \
   /        \   Integration Tests (20%)
  /----------\  Rust↔Python, API endpoints
 /            \
/              \ Unit Tests (75%)
----------------\ Individual functions, classes
```

---

## Test Organization

### Directory Structure

```
nglab/
├── rust/
│   └── tests/              # Rust integration tests
│       ├── test_orderbook.rs
│       ├── test_gym.rs
│       └── test_moon.rs
├── python/
│   └── tests/
│       ├── conftest.py     # Shared fixtures
│       ├── fixtures/       # Test data factories
│       │   ├── env_fixtures.py
│       │   ├── model_fixtures.py
│       │   └── hpo_fixtures.py
│       ├── unit/           # Unit tests
│       │   ├── test_vae.py
│       │   ├── test_mamba.py
│       │   └── test_orderbook.py
│       ├── integration/    # Integration tests
│       │   └── test_rust_bindings.py
│       └── e2e/            # End-to-end tests
│           └── test_training_pipeline.py
└── typescript/
    └── src/test/           # Frontend tests
        ├── components/     # Component unit tests
        ├── hooks/          # Hook tests
        └── e2e/            # Playwright E2E tests
```

### Naming Conventions

| Language       | Pattern                            | Example                           |
| -------------- | ---------------------------------- | --------------------------------- |
| **Rust**       | `test_<function>_<scenario>`       | `test_add_order_limit_buy`        |
| **Python**     | `test_<class>_<method>_<scenario>` | `test_vae_forward_batch_size_one` |
| **TypeScript** | `<Component>.test.tsx`             | `PriceChart.test.tsx`             |

---

## Running Tests

### All Tests

```bash
just test           # Runs Rust, Python, TypeScript tests
```

### By Language

```bash
just test-rust      # Rust tests
just test-python    # Python tests
just test-typescript # TypeScript tests
```

### With Coverage

```bash
just coverage
# Output:
#   coverage/rust/     - Rust HTML coverage
#   coverage/python/   - Python HTML coverage
```

### Specific Tests

```bash
# Rust - single test
cargo test test_orderbook_matching -- --exact

# Rust - single module
cargo test orderbook::

# Python - single file
cd python && pytest tests/unit/test_vae.py -v

# Python - single test function
cd python && pytest tests/unit/test_vae.py::test_forward -v

# Python - by marker
cd python && pytest -m "slow" -v

# TypeScript - single test
cd typescript && npm test -- --grep "PriceChart"
```

### Watch Mode

```bash
# Rust - rerun on file change
cargo watch -x test

# Python - rerun on file change
cd python && pytest-watch

# TypeScript - watch mode
cd typescript && npm test -- --watch
```

---

## Coverage Requirements

### Targets by Component

| Component                 | Coverage Target | Enforcement  |
| ------------------------- | --------------- | ------------ |
| **Rust Core**             | 80%             | CI blocks PR |
| **Python Models**         | 70%             | CI warning   |
| **Python Pipeline**       | 70%             | CI warning   |
| **TypeScript Components** | 60%             | CI warning   |
| **Critical Paths**        | 95%             | CI blocks PR |

### Critical Paths (Must be 95%+)

- `rust/src/simulation/orderbook.rs` - Order matching
- `rust/src/simulation/gym.rs` - Environment step
- `python/src/models/deep/autoencoders/vae.py` - VAE forward/loss
- `python/src/pipeline/hpo/dehb.py` - HPO core logic

### Generating Coverage Reports

```bash
# Rust (requires cargo-tarpaulin)
cargo tarpaulin --out Html --output-dir coverage/rust

# Python
cd python && pytest --cov=src --cov-report=html:coverage

# TypeScript
cd typescript && npm run test:coverage
```

---

## Rust Testing

### Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_order_creation() {
        let order = Order::new(OrderType::Limit, Side::Buy, 100.0, 1.0);
        assert_eq!(order.price, 100.0);
        assert_eq!(order.quantity, 1.0);
    }

    #[test]
    #[should_panic(expected = "negative quantity")]
    fn test_order_negative_quantity_panics() {
        Order::new(OrderType::Limit, Side::Buy, 100.0, -1.0);
    }
}
```

### Property-Based Tests (proptest)

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_orderbook_never_crosses(bids in any::<Vec<f64>>(), asks in any::<Vec<f64>>()) {
        let mut ob = OrderBook::new();
        // ... setup
        prop_assert!(ob.best_bid().unwrap_or(0.0) < ob.best_ask().unwrap_or(f64::MAX));
    }
}
```

### Benchmark Tests (Criterion)

```rust
// benches/orderbook_bench.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn orderbook_insert_benchmark(c: &mut Criterion) {
    c.bench_function("insert 1000 orders", |b| {
        b.iter(|| {
            let mut ob = OrderBook::new();
            for i in 0..1000 {
                ob.add_order(black_box(Order::new(...)));
            }
        })
    });
}

criterion_group!(benches, orderbook_insert_benchmark);
criterion_main!(benches);
```

---

## Python Testing

### Fixtures (conftest.py)

```python
# python/tests/conftest.py
import pytest
import torch
import numpy as np

@pytest.fixture
def sample_prices():
    """Generate 100 days of random prices."""
    np.random.seed(42)
    returns = np.random.randn(100) * 0.02
    return 100 * np.exp(np.cumsum(returns))

@pytest.fixture
def mock_env():
    """Create a mock TradingEnv."""
    try:
        from nglab import TradingEnv
        return TradingEnv()
    except ImportError:
        pytest.skip("Rust extension not available")

@pytest.fixture
def device():
    """Return available torch device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### Unit Test Example

```python
# tests/unit/test_vae.py
import pytest
import torch
from models.deep.autoencoders.vae import VAE

class TestVAE:
    def test_forward_shape(self, device):
        model = VAE(input_dim=60, latent_dim=8).to(device)
        x = torch.randn(32, 60).to(device)

        recon, mu, logvar = model(x)

        assert recon.shape == (32, 60)
        assert mu.shape == (32, 8)
        assert logvar.shape == (32, 8)

    def test_loss_positive(self, device):
        model = VAE(input_dim=60, latent_dim=8).to(device)
        x = torch.randn(32, 60).to(device)

        recon, mu, logvar = model(x)
        loss = model.loss_function(recon, x, mu, logvar)

        assert loss > 0
```

### Property-Based Tests (Hypothesis)

```python
from hypothesis import given, strategies as st

@given(
    batch_size=st.integers(min_value=1, max_value=128),
    seq_len=st.integers(min_value=10, max_value=1000),
)
def test_model_handles_variable_inputs(batch_size, seq_len):
    model = SimpleModel(input_dim=seq_len)
    x = torch.randn(batch_size, seq_len)

    output = model(x)

    assert output.shape[0] == batch_size
    assert not torch.isnan(output).any()
```

### Markers

```python
# pytest.ini
[pytest]
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    gpu: marks tests requiring GPU
    integration: marks integration tests

# Usage in test
@pytest.mark.slow
@pytest.mark.gpu
def test_full_training_loop():
    ...
```

---

## TypeScript Testing

### Component Tests (Vitest)

```typescript
// src/test/components/PriceChart.test.tsx
import { render, screen } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { PriceChart } from '../../components/PriceChart';

describe('PriceChart', () => {
  it('renders without crashing', () => {
    render(<PriceChart data={[]} />);
    expect(screen.getByTestId('price-chart')).toBeInTheDocument();
  });

  it('displays loading state when no data', () => {
    render(<PriceChart data={[]} loading={true} />);
    expect(screen.getByText('Loading...')).toBeInTheDocument();
  });
});
```

### Hook Tests

```typescript
// src/test/hooks/useArena.test.ts
import { renderHook, act } from "@testing-library/react";
import { useArena } from "../../hooks/useArena";
import { vi } from "vitest";

// Mock Tauri
vi.mock("@tauri-apps/api/event", () => ({
  listen: vi.fn().mockResolvedValue(() => {}),
}));

describe("useArena", () => {
  it("initializes with default state", () => {
    const { result } = renderHook(() => useArena());

    expect(result.current.isRunning).toBe(false);
    expect(result.current.stepInfo).toBeNull();
  });
});
```

### Snapshot Testing

```typescript
it('matches snapshot', () => {
  const { container } = render(<OrderBookTable bids={[]} asks={[]} />);
  expect(container).toMatchSnapshot();
});
```

---

## Integration Testing

### Rust ↔ Python Integration

```python
# tests/integration/test_rust_bindings.py
import pytest

def test_orderbook_roundtrip():
    """Test creating and using OrderBook from Python."""
    try:
        from nglab import OrderBook, Order, OrderType
    except ImportError:
        pytest.skip("Rust extension not built")

    ob = OrderBook()
    order = Order(OrderType.Limit, "BUY", 100.0, 1.0, 0)

    trades = ob.add_order(order)

    assert ob.best_bid() == 100.0
    assert len(trades) == 0  # No matching asks

def test_trading_env_step():
    """Test TradingEnv step cycle."""
    try:
        from nglab import TradingEnv
    except ImportError:
        pytest.skip("Rust extension not built")

    env = TradingEnv()
    obs, info = env.reset()

    obs2, reward, done, truncated, info = env.step(1)  # Buy

    assert obs2.shape == obs.shape
    assert isinstance(reward, float)
```

### API Integration

```python
# tests/integration/test_api.py
import pytest
from fastapi.testclient import TestClient
from api.app import app

@pytest.fixture
def client():
    return TestClient(app)

def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict_endpoint(client):
    response = client.post("/predict", json={"prices": [100, 101, 102]})
    assert response.status_code == 200
    assert "prediction" in response.json()
```

---

## Performance Benchmarks

### Rust Benchmarks

```bash
cargo bench                    # Run all benchmarks
cargo bench orderbook          # Run specific benchmark
```

### Python Benchmarks (pytest-benchmark)

```python
# tests/benchmarks/test_model_performance.py
def test_vae_forward_speed(benchmark):
    model = VAE(input_dim=60, latent_dim=8)
    x = torch.randn(64, 60)

    result = benchmark(model.forward, x)

    assert result[0].shape == (64, 60)

def test_environment_step_speed(benchmark):
    env = TradingEnv()
    env.reset()

    benchmark(env.step, 1)
```

### Performance Regression Detection

```yaml
# CI will fail if performance regresses > 10%
- name: Performance Gate
  run: |
    cargo bench -- --save-baseline current
    cargo bench -- --baseline main --threshold 1.10
```

---

## Mocking Strategies

### Rust Mocking

```rust
// Use trait objects for mockable dependencies
trait DataSource {
    fn fetch_prices(&self) -> Vec<f64>;
}

struct MockDataSource {
    prices: Vec<f64>,
}

impl DataSource for MockDataSource {
    fn fetch_prices(&self) -> Vec<f64> {
        self.prices.clone()
    }
}
```

### Python Mocking

```python
from unittest.mock import Mock, patch, AsyncMock

def test_with_mock_api():
    with patch('requests.get') as mock_get:
        mock_get.return_value.json.return_value = {"price": 100.0}

        result = fetch_price("BTC")

        assert result == 100.0
        mock_get.assert_called_once()

@pytest.mark.asyncio
async def test_async_function():
    with patch('aiohttp.ClientSession.get', new_callable=AsyncMock) as mock:
        mock.return_value.__aenter__.return_value.json = AsyncMock(return_value={})

        result = await async_fetch()

        assert result == {}
```

### TypeScript Mocking

```typescript
// Mock Tauri invoke
vi.mock("@tauri-apps/api/core", () => ({
  invoke: vi.fn().mockResolvedValue({ success: true }),
}));

// Mock React hooks
vi.mock("react", async () => {
  const actual = await vi.importActual("react");
  return {
    ...actual,
    useContext: vi.fn().mockReturnValue({ user: mockUser }),
  };
});
```

---

## CI/CD Integration

### GitHub Actions Test Matrix

```yaml
# .github/workflows/test.yml
jobs:
  test-rust:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - run: cargo test --all-features

  test-python:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
      - run: cd python && uv sync && pytest

  test-typescript:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with: { node-version: 20 }
      - run: cd typescript && npm ci && npm test
```

### Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: rust-tests
        name: Rust Tests
        entry: cargo test --lib
        language: system
        pass_filenames: false
        files: \.rs$

      - id: python-tests
        name: Python Tests
        entry: bash -c 'cd python && pytest tests/unit -q'
        language: system
        pass_filenames: false
        files: \.py$
```

---

## Test Data Management

### Fixtures Factory Pattern

```python
# tests/fixtures/model_fixtures.py
import torch

def create_sample_batch(batch_size=32, seq_len=60, features=1):
    """Factory for creating sample input batches."""
    return torch.randn(batch_size, seq_len, features)

def create_price_series(n_days=100, start_price=100.0, volatility=0.02):
    """Factory for creating realistic price series."""
    returns = np.random.normal(0, volatility, n_days)
    return start_price * np.exp(np.cumsum(returns))
```

### Test Data Files

```python
# Load from fixtures
@pytest.fixture
def sample_orderbook_data():
    with open("tests/fixtures/orderbook_snapshot.json") as f:
        return json.load(f)
```

### Database Test Isolation

```python
@pytest.fixture
def test_db(tmp_path):
    """Create isolated test database."""
    db_path = tmp_path / "test.db"
    init_database(db_path)
    yield db_path
    db_path.unlink()  # Cleanup
```

---

## Best Practices Checklist

- [ ] Every new feature has unit tests
- [ ] Bug fixes include regression tests
- [ ] Critical paths have >95% coverage
- [ ] Tests are deterministic (no flakiness)
- [ ] Test names describe expected behavior
- [ ] Slow tests are marked `@pytest.mark.slow`
- [ ] GPU tests are marked `@pytest.mark.gpu`
- [ ] Mocks are cleaned up after tests
- [ ] No hardcoded file paths in tests

---

**Remember**: Tests are the safety net that lets us refactor with confidence.
