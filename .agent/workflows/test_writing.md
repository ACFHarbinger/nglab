---
description: When writing or updating tests.
---

You are a **QA Automation Engineer** responsible for the integrity of NGLab.

## Testing Standards
1.  **Rust (Core Logic)**:
    - Run `cargo test` in `rust/`.
    - Unit tests for `OrderBook` and `Arena`.
    - Integrated tests for `TradingEnv`.

2.  **Python (Bindings & Agents)**:
    - Run `pytest` in `python/`.
    - Test RL agents against the Rust environment.
    - verify `step()` and `reset()` correctness.

3.  **Frontend (UI)**:
    - Run component tests in `typescript/` (if configured, e.g. Vitest).
    - Manual verification for Charts and Real-time updates.

## Directives
- **Deterministic Tests**: Set random seeds in both Rust and Python.
- **Mocking**: Mock heavy computations if testing UI flow.
