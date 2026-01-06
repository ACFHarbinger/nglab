# NGLab - Next Gen Laboratory

## Overview
NGLab is a Multimodal Deep Reinforcement Learning bot designed for financial trading. It leverages a high-performance Rust backend for simulation and a Python environment for training and strategy implementation.

## Tech Stack
- **Core Logic & Simulation**: Rust (`rust/`)
  - `nglab` crate: Handles the `Arena`, `OrderBook`, `PolymarketArena`, and `TradingEnv`.
  - Exposed to Python via PyO3.
- **Training & Bindings**: Python (`python/`)
  - Uses `gym` compatible interface from Rust.
  - Deep Learning support (PyTorch/TensorFlow implied by requirements).
- **Frontend / Dashboard**: Tauri 2.0 + React + TypeScript (`typescript/`)
  - **Framework**: Tauri 2.0 (Rust backend + Web frontend)
  - **UI**: React 19, Tailwind CSS, Shadcn UI principles.
  - **Charts**: `lightweight-charts` for high-performance financial plotting.
  - **State**: `useArena` hook for real-time event streaming from Tauri.

## Architecture
- **Rust Backend (`typescript/src-tauri`)**:
  - Contains `ArenaState` wrapping the `TradingEnv` in a Mutex.
  - Runs a dedicated Tokio task for the simulation loop.
  - Emits `arena-update` events with `StepInfo` and `OrderBook` snapshots.
- **Frontend (`typescript/src`)**:
  - Listens for `arena-update` and renders Price Charts and Order Book visualization.

## Development
- **Rust**: `cargo check`, `cargo build`
- **Tauri**: `pnpm tauri dev` (requires system dependencies)
