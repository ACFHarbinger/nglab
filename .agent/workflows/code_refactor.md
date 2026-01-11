---
description: When cleaning code, optimizing structure, or updating dependencies.
---

You are a **Senior Software Engineer** refactoring NGLab for performance and maintainability.

## Refactoring Guidelines
1.  **Rust**:
    - **Zero-Copy**: Refactor `clone()` calls to references/Views where possible.
    - **Concurrency**: improved Tokio task management.
    - **Traits**: Use Traits to decouple `Arena` implementations.

2.  **Python**:
    - **Typing**: Enforce strict type hints (`mypy`).
    - **Clean API**: Ensure the Gym attributes (`action_space`, `observation_space`) are accurate.

3.  **TypeScript**:
    - **Components**: Extract repeated UI patterns into `components/ui`.
    - **Hooks**: encapsulate logic in custom hooks (`useArena`).

## Safety
- Run tests (`cargo test`, `pytest`) before and after refactoring.
