# NGLab Task Automation — Root Justfile
# https://github.com/casey/just
#
# Recipes are organised into per-domain sub-modules under tools/. Invoke a
# sub-module recipe directly (e.g. `just build::rust`, `just crypto::run`), or
# use the root shorthands below.

set shell := ["bash", "-c"]
set unstable := true

# --- Sub-module declarations (imported from tools/) ---

mod helper   "tools/helper/justfile"
mod dev      "tools/dev/justfile"
mod build    "tools/build/justfile"
mod test     "tools/test/justfile"
mod quality  "tools/quality/justfile"
mod run      "tools/run/justfile"
mod docs     "tools/docs/justfile"
mod bench    "tools/bench/justfile"
mod crypto   "tools/crypto/justfile"
mod hft      "tools/hft/justfile"

# --- Default target ---

default: help

# List all commands across every sub-module
help:
    @just helper::help

# Project statistics
stats:
    @just helper::stats

# --- Setup & maintenance (→ tools/dev) ---

# Set up the full development environment
setup:
    @just dev::setup

# Update all dependencies
update:
    @just dev::update

# Install development tools
install-tools:
    @just dev::install-tools

# Run pre-commit hooks
pre-commit:
    @just dev::pre-commit

# Clean build artifacts
clean:
    @just dev::clean

# Deep clean (incl. node_modules)
clean-all:
    @just dev::clean-all

# --- Build (→ tools/build) ---

# Build every tier
build-all:
    @just build::all

# --- Test (→ tools/test) ---

# Run every tier's tests
test-all:
    @just test::all

# Coverage reports
coverage:
    @just test::coverage

# --- Quality (→ tools/quality) ---

# Format all code
fmt:
    @just quality::fmt

# Lint all code
lint:
    @just quality::lint

# Auto-fix lint issues
fix:
    @just quality::fix

# Audit dependencies
audit:
    @just quality::audit

# Detect undefined behavior (Rust miri)
detect-ub:
    @just quality::detect-ub

# Quality gate: lint + all tests
check: lint test-all
    @echo "✅ Code quality check passed!"

# Format + lint + test, ready to push
pre-push: fmt lint test-all
    @echo "✅ Ready to push!"

# --- Run (→ tools/run) ---
# Note: `dev`, `docs`, and `bench` are sub-module names, so the root shorthands
# below use distinct names (serve / docs-gen / bench-run). The full sets are
# `just run::dev`, `just docs::all`, `just bench::rust`.

# Start the Tauri dev server
serve:
    @just run::dev

# Train an RL agent
train MODEL="ppo":
    @just run::train {{MODEL}}

# Evaluate a trained agent
evaluate CHECKPOINT:
    @just run::evaluate {{CHECKPOINT}}

# Start the Jupyter notebook server
notebook:
    @just run::notebook

# Generate mock trading data
seed-data assets="BTC,ETH,SOL" days="7":
    @just run::seed-data {{assets}} {{days}}

# --- Docs & bench ---

# Generate consolidated API docs
docs-gen:
    @just docs::all

# Run Rust benchmarks
bench-run:
    @just bench::rust

# --- Polyglot tiers (shorthands; see crypto::/hft:: for the full set) ---

# Build + run the Go crypto daemon on a loopback port
crypto-run PORT="54321":
    @just crypto::run {{PORT}}

# Build the C++ HFT daemon
hft-build:
    @just hft::build
