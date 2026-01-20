#!/bin/bash
# NGLab Build Artifact Cleanup Script
# 
# Removes build artifacts, caches, and temporary files across the entire 
# project (Rust, Python, TypeScript) to reclaim disk space and ensure 
# clean builds.
#
# Sections:
#   1. Rust build artifacts (target/)
#   2. Python cache files (__pycache__, .pyc, etc.)
#   3. TypeScript/Node.js build artifacts (dist/, src-tauri/target/)
#   4. Test and coverage artifacts (.coverage, htmlcov)
#   5. Jupyter notebook checkpoints
#   6. OS and editor artifacts (.DS_Store, Thumbs.db, swap files)
#
# Usage: ./scripts/cleanup.sh

set -e

echo "======================================"
echo "NGLab Build Artifact Cleanup"
echo "======================================"
echo ""

# Get initial disk usage
INITIAL_USAGE=$(du -sh . 2>/dev/null | cut -f1)
echo "Initial directory size: $INITIAL_USAGE"
echo ""

# Function to show disk space saved
show_savings() {
    CURRENT_USAGE=$(du -sh . 2>/dev/null | cut -f1)
    echo "Current directory size: $CURRENT_USAGE"
}

# Clean Rust build artifacts
echo "[1/6] Cleaning Rust build artifacts..."
if [ -f "Cargo.toml" ]; then
    cargo clean 2>/dev/null || true
    echo "  ✓ Rust target/ directory cleaned"
else
    echo "  ⊘ No Cargo.toml found, skipping"
fi
echo ""

# Clean Python cache
echo "[2/6] Cleaning Python cache files..."
PYC_COUNT=$(find . -type f -name "*.pyc" 2>/dev/null | wc -l)
PYCACHE_COUNT=$(find . -type d -name "__pycache__" 2>/dev/null | wc -l)

find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type f -name "*.pyo" -delete 2>/dev/null || true

echo "  ✓ Removed $PYC_COUNT .pyc files"
echo "  ✓ Removed $PYCACHE_COUNT __pycache__ directories"
echo "  ✓ Removed .pytest_cache, .ruff_cache, .mypy_cache"
echo ""

# Clean TypeScript/Node build artifacts
echo "[3/6] Cleaning TypeScript/Node.js build artifacts..."
if [ -d "typescript" ]; then
    cd typescript

    if [ -d "dist" ]; then
        rm -rf dist
        echo "  ✓ Removed dist/ directory"
    fi

    if [ -d "node_modules" ]; then
        NODE_SIZE=$(du -sh node_modules 2>/dev/null | cut -f1)
        echo "  ℹ node_modules/ size: $NODE_SIZE (preserved)"
        echo "    Run 'cd typescript && rm -rf node_modules' to remove"
    fi

    if [ -d "src-tauri/target" ]; then
        rm -rf src-tauri/target
        echo "  ✓ Removed src-tauri/target/ directory"
    fi

    cd ..
else
    echo "  ⊘ No typescript/ directory found"
fi
echo ""

# Clean pytest and coverage files
echo "[4/6] Cleaning test and coverage artifacts..."
find . -type f -name ".coverage" -delete 2>/dev/null || true
find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name ".tox" -exec rm -rf {} + 2>/dev/null || true
echo "  ✓ Removed .coverage, htmlcov, .tox"
echo ""

# Clean Jupyter notebook checkpoints
echo "[5/6] Cleaning Jupyter notebook checkpoints..."
CHECKPOINT_COUNT=$(find . -type d -name ".ipynb_checkpoints" 2>/dev/null | wc -l)
find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
echo "  ✓ Removed $CHECKPOINT_COUNT .ipynb_checkpoints directories"
echo ""

# Clean macOS and editor artifacts
echo "[6/6] Cleaning OS and editor artifacts..."
find . -name ".DS_Store" -delete 2>/dev/null || true
find . -name "Thumbs.db" -delete 2>/dev/null || true
find . -name "*.swp" -delete 2>/dev/null || true
find . -name "*.swo" -delete 2>/dev/null || true
find . -name "*~" -delete 2>/dev/null || true
echo "  ✓ Removed .DS_Store, Thumbs.db, swap files"
echo ""

# Optional: Clean Cargo cache (commented out by default)
# Uncomment the following lines to clean Cargo cache as well
# echo "[Optional] Cleaning Cargo cache..."
# if command -v cargo-cache &> /dev/null; then
#     cargo cache --autoclean
#     echo "  ✓ Cargo cache cleaned"
# else
#     echo "  ⊘ cargo-cache not installed. Install with: cargo install cargo-cache"
# fi
# echo ""

# Show final results
echo "======================================"
echo "Cleanup Complete!"
echo "======================================"
show_savings
echo ""
echo "Disk space recovered. Build artifacts removed."
echo ""
echo "To also remove node_modules (can be large):"
echo "  cd typescript && rm -rf node_modules"
echo ""
echo "To reinstall dependencies after cleanup:"
echo "  Rust:       cargo build"
echo "  Python:     uv sync  (or pip install -e .)"
echo "  TypeScript: cd typescript && npm ci"
