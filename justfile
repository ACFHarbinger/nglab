# NGLab Task Automation
# https://github.com/casey/just

# Default recipe (list all commands)
default:
    @just --list

# Setup development environment
setup:
    @echo "🚀 Setting up NGLab development environment..."
    @echo ""
    @echo "[1/5] Installing Rust toolchain..."
    rustup update stable
    rustup component add rustfmt clippy
    @echo ""
    @echo "[2/5] Installing Python dependencies..."
    cd python && uv sync
    @echo ""
    @echo "[3/5] Installing TypeScript dependencies..."
    cd typescript && npm ci
    @echo ""
    @echo "[4/5] Installing pre-commit hooks..."
    pip install pre-commit || true
    pre-commit install || echo "⚠️  pre-commit not available, skipping hooks"
    @echo ""
    @echo "[5/5] Installing additional tools..."
    cargo install cargo-cache cargo-audit || true
    @echo ""
    @echo "✅ Setup complete! Run 'just test' to verify installation."

# Run all tests
test:
    @echo "🧪 Running all tests..."
    @just test-rust
    @just test-python
    @just test-typescript

# Run Rust tests
test-rust:
    @echo "Testing Rust codebase..."
    cargo test --all-features --workspace

# Run Python tests
test-python:
    @echo "Testing Python codebase..."
    cd python && pytest -v --tb=short

# Run TypeScript tests
test-typescript:
    @echo "Testing TypeScript codebase..."
    cd typescript && npm test

# Run integration tests
test-integration:
    @echo "Running integration tests..."
    cd python && pytest tests/test_integration.py -v

# Format all code
fmt:
    @echo "🎨 Formatting all code..."
    @just fmt-rust
    @just fmt-python
    @just fmt-typescript

# Format Rust code
fmt-rust:
    @echo "Formatting Rust..."
    cargo fmt --all

# Format Python code
fmt-python:
    @echo "Formatting Python..."
    cd python && ruff format .

# Format TypeScript code
fmt-typescript:
    @echo "Formatting TypeScript..."
    cd typescript && npm run format || npx prettier --write "src/**/*.{ts,tsx}"

# Lint all code
lint:
    @echo "🔍 Linting all code..."
    @just lint-rust
    @just lint-python
    @just lint-typescript

# Lint Rust code
lint-rust:
    @echo "Linting Rust..."
    cargo clippy --all-features --workspace -- -D warnings

# Lint Python code
lint-python:
    @echo "Linting Python..."
    cd python && ruff check .

# Lint TypeScript code
lint-typescript:
    @echo "Linting TypeScript..."
    cd typescript && npm run lint || npx eslint "src/**/*.{ts,tsx}"

# Fix linting issues automatically
fix:
    @echo "🔧 Auto-fixing linting issues..."
    cargo clippy --all-features --workspace --fix --allow-dirty --allow-staged
    cd python && ruff check --fix .
    cd typescript && npm run lint:fix || npx eslint "src/**/*.{ts,tsx}" --fix

# Detect cases of undefined behavior (e.g., out-of-bounds memory access)
detect-undefined-behavior:
    @echo "🔍 Detecting undefined behavior..."
    cargo miri test

# Build everything
build:
    @echo "🔨 Building all components..."
    @just build-rust
    @just build-python
    @just build-typescript

# Build Rust (release mode)
build-rust:
    @echo "Building Rust..."
    cargo build --release

# Build Python package
build-python:
    @echo "Building Python package..."
    cd python && maturin develop --release

# Build TypeScript/Tauri app
build-typescript:
    @echo "Building TypeScript/Tauri..."
    cd typescript && npm run build

# Build Tauri desktop app
build-tauri:
    @echo "Building Tauri desktop application..."
    cd typescript && npm run tauri build

# Clean build artifacts
clean:
    @echo "🧹 Cleaning build artifacts..."
    @./script/cleanup.sh

# Deep clean (including node_modules)
clean-all: clean
    @echo "Deep cleaning node_modules..."
    rm -rf typescript/node_modules
    @echo "✅ Deep clean complete"

# Reset all credentials (deletes local databases)
reset-credentials:
    @echo "🧹 Resetting all credentials..."
    @rm -f assets/secrets/*.db
    @echo "✅ Local databases deleted."

# Run development server for Tauri app
dev:
    @echo "🚀 Starting Tauri development server..."
    cd typescript && npm run tauri dev

# Run benchmarks
bench:
    @echo "📊 Running benchmarks..."
    cargo bench

# Run Python training
train MODEL="ppo":
    @echo "🏋️  Training {{MODEL}} agent..."
    cd python && python -m python.src.pipeline.train model={{MODEL}}

# Evaluate trained agent
evaluate CHECKPOINT:
    @echo "📈 Evaluating agent from {{CHECKPOINT}}..."
    cd python && python -m python.src.pipeline.evaluate_agents --checkpoint={{CHECKPOINT}}

# Update all dependencies
update:
    @echo "⬆️  Updating dependencies..."
    cargo update
    cd python && uv sync --upgrade
    cd typescript && npm update
    @echo "✅ Dependencies updated. Run 'just test' to verify."

# Check for security vulnerabilities
audit:
    @echo "🔒 Auditing dependencies for vulnerabilities..."
    cargo audit
    cd python && pip-audit || echo "⚠️  pip-audit not installed: pip install pip-audit"
    cd typescript && npm audit

# Generate documentation
docs:
    @echo "📚 Generating consolidated documentation..."
    @mkdir -p docs/rust docs/python
    cargo doc --no-deps
    cd python && pdoc --html --output-dir ../docs/python python.src
    @echo "✅ Documentation generated in docs/ (Rust: target/doc, Python: docs/python)"

# Generate mock data for the simulation
seed-data assets="BTC,ETH,SOL" days="7":
    @echo "🌱 Generating mock trading data for {{assets}}..."
    mkdir -p assets/data
    cd python && uv run python ../script/seed_data.py --assets {{assets}} --days {{days}} --output ../assets/data/
    @echo "✅ Mock data generated in assets/data/"

# Check code quality (lint + test)
check: lint test
    @echo "✅ Code quality check passed!"

# Run pre-commit hooks manually
pre-commit:
    @echo "🔍 Running pre-commit hooks..."
    pre-commit run --all-files

# Initialize secrets baseline for detect-secrets
init-secrets:
    @echo "🔐 Initializing secrets baseline..."
    detect-secrets scan > .secrets.baseline
    @echo "✅ Secrets baseline created"

# Profile Rust code
profile BENCH="orderbook":
    @echo "🔬 Profiling {{BENCH}} benchmark..."
    cargo flamegraph --bench {{BENCH}}

# Run Python code with profiler
profile-python SCRIPT:
    @echo "🔬 Profiling Python script: {{SCRIPT}}..."
    py-spy record -o profile.svg -- python {{SCRIPT}}

# Watch and rebuild on file changes (Rust)
watch:
    @echo "👀 Watching for Rust file changes..."
    cargo watch -x check -x test

# Start Jupyter notebook server
notebook:
    @echo "📓 Starting Jupyter notebook server..."
    cd python && jupyter notebook

# Run code coverage
coverage:
    @echo "📊 Generating code coverage reports..."
    @echo "Rust coverage..."
    cargo tarpaulin --out Html --output-dir coverage/rust || echo "⚠️  cargo-tarpaulin not installed"
    @echo "Python coverage..."
    cd python && pytest --cov=python.src --cov-report=html:../coverage/python
    @echo "✅ Coverage reports in coverage/ directory"

# Install development tools
install-tools:
    @echo "🔧 Installing development tools..."
    cargo install cargo-watch cargo-tarpaulin cargo-cache cargo-audit cargo-flamegraph
    pip install pre-commit pytest-cov py-spy pip-audit detect-secrets
    @echo "✅ Tools installed"

# Quick check before commit
pre-push: fmt lint test
    @echo "✅ Ready to push!"

# Show project statistics
stats:
    @echo "📊 NGLab Project Statistics"
    @echo "==========================="
    @echo ""
    @echo "Lines of code:"
    @tokei
    @echo ""
    @echo "Rust crates:"
    @cargo tree --depth 1
    @echo ""
    @echo "Build artifacts size:"
    @du -sh target/ 2>/dev/null || echo "No Rust build artifacts"
    @du -sh python/.venv/ 2>/dev/null || echo "No Python virtualenv"
    @du -sh typescript/node_modules/ 2>/dev/null || echo "No Node modules"
