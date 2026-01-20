SHELL := /bin/bash

# Colors for terminal output
BLUE         := \033[0;34m
CYAN         := \033[0;36m
GREEN        := \033[0;32m
YELLOW       := \033[1;33m
RED          := \033[0;31m
MAGENTA      := \033[0;35m
NC           := \033[0m # No Color

# --- Macros ---
define print_header
	@echo -e "$(BLUE)╔══════════════════════════════════════════════════════════════════════╗$(NC)"
	@echo -e "$(BLUE)║                  NGLab Trading Framework Control                     ║$(NC)"
	@echo -e "$(BLUE)╚══════════════════════════════════════════════════════════════════════╝$(NC)"
endef

.PHONY: help test test-rust test-python build build-rust build-python lint fmt clean dev install-hooks run-tauri

# Color codes
GREEN := \033[0;32m
BLUE := \033[0;34m
YELLOW := \033[0;33m
CYAN := \033[0;36m
RED := \033[0;31m
RESET := \033[0m
BOLD := \033[1m

# Default target
help:
	@echo "$(CYAN)$(BOLD)NGLab Development Makefile$(RESET)"
	@echo ""
	@echo "$(BOLD)Available targets:$(RESET)"
	@echo "  $(GREEN)make test$(RESET)          	- Run all tests (Python + Rust)"
	@echo "  $(GREEN)make test-python$(RESET)   	- Run Python tests with pytest"
	@echo "  $(GREEN)make test-rust$(RESET)     	- Run Rust tests with cargo"
	@echo "  $(BLUE)make build$(RESET)         		- Build all components"
	@echo "  $(BLUE)make build-rust$(RESET)    		- Build Rust bindings with maturin"
	@echo "  $(BLUE)make build-python$(RESET)  		- Install Python package"
	@echo "  $(YELLOW)make lint$(RESET)          	- Run all linters (ruff + rustfmt + eslint)"
	@echo "  $(YELLOW)make fmt$(RESET)           	- Auto-format all code"
	@echo "  $(RED)make clean$(RESET)         		- Clean build artifacts"
	@echo "  $(CYAN)make dev$(RESET)           		- Setup development environment"
	@echo "  $(CYAN)make install-hooks$(RESET) 		- Install pre-commit hooks"
	@echo "  $(CYAN)make run-tauri$(RESET)     		- Run Tauri development server"

# Testing
test: test-python test-rust
	@echo "$(GREEN)✅ All tests passed!$(RESET)"

test-python:
	@echo "$(BLUE)🐍 Running Python tests...$(RESET)"
	pytest python/tests/ -v

test-python-gpu:
	@echo "$(BLUE)🚀 Running Python tests (including GPU tests)...$(RESET)"
	pytest python/tests/ -v -m gpu

test-rust:
	@echo "$(BLUE)🦀 Running Rust tests...$(RESET)"
	cargo test --workspace

# Building
build: build-rust build-python
	@echo "$(GREEN)✅ Build complete!$(RESET)"

build-rust:
	@echo "$(CYAN)🔨 Building Rust bindings with maturin...$(RESET)"
	cd rust && maturin develop --release

build-python:
	@echo "$(CYAN)📦 Installing Python package...$(RESET)"
	pip install -e .

# Linting
lint: lint-python lint-rust lint-typescript
	@echo "$(GREEN)✅ Linting complete!$(RESET)"

lint-python:
	@echo "$(YELLOW)🔍 Linting Python code...$(RESET)"
	ruff check python/

lint-rust:
	@echo "$(YELLOW)🔍 Checking Rust code formatting...$(RESET)"
	cargo fmt --check
	cargo clippy --all-targets --all-features -- -D warnings

lint-typescript:
	@echo "$(YELLOW)🔍 Linting TypeScript code...$(RESET)"
	cd typescript && npm run lint

# Formatting
fmt: fmt-python fmt-rust fmt-typescript
	@echo "$(GREEN)✅ Formatting complete!$(RESET)"

fmt-python:
	@echo "$(CYAN)✨ Formatting Python code...$(RESET)"
	ruff format python/
	ruff check --fix python/

fmt-rust:
	@echo "$(CYAN)✨ Formatting Rust code...$(RESET)"
	cargo fmt

fmt-typescript:
	@echo "$(CYAN)✨ Formatting TypeScript code...$(RESET)"
	cd typescript && npm run format

# Cleanup
clean:
	@echo "$(RED)🧹 Cleaning build artifacts...$(RESET)"
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf target/
	rm -rf .pytest_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete
	@echo "$(GREEN)✅ Cleanup complete!$(RESET)"

# Development setup
dev: install-hooks
	@echo "$(CYAN)$(BOLD)🚀 Setting up development environment...$(RESET)"
	@echo "$(BLUE)📦 Installing Python dependencies...$(RESET)"
	pip install -e ".[dev]"
	@echo "$(BLUE)🔨 Building Rust bindings...$(RESET)"
	cd rust && maturin develop
	@echo "$(BLUE)📦 Installing TypeScript dependencies...$(RESET)"
	cd typescript && npm install
	@echo "$(GREEN)✅ Development environment ready!$(RESET)"

install-hooks:
	@echo "$(CYAN)🪝 Installing pre-commit hooks...$(RESET)"
	pre-commit install
	@echo "$(GREEN)✅ Pre-commit hooks installed!$(RESET)"

# Tauri development
run-tauri:
	@echo "$(CYAN)🖥️  Starting Tauri development server...$(RESET)"
	cd typescript && npm run tauri dev

# Benchmarking
benchmark:
	@echo "$(YELLOW)⚡ Running Rust benchmarks...$(RESET)"
	cargo bench

# Coverage
coverage:
	@echo "$(BLUE)📊 Running tests with coverage...$(RESET)"
	pytest python/tests/ --cov=python/src --cov-report=html --cov-report=term
	@echo "$(GREEN)Coverage report generated in htmlcov/$(RESET)"
