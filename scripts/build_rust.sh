#!/bin/bash
# NGLab Rust Build Script
# 
# This script builds the Rust package using Maturin and prepares it for 
# Python integration. It executes 'maturin build' with the --release flag.
#
# Usage: ./scripts/build_rust.sh

cd rust
maturin build --release