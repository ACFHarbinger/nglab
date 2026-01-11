# Refactoring Safety Prompt

**Intent:** Safely modify core logic using the Constraint pattern in Image-Toolkit.

## The Prompt

I need to modify the following core component: `[INSERT COMPONENT/FILE]`.

**Current Goal:** [Brief description of change, e.g., "Add new image filter to Rust core"].

**Strict Constraints:**
1.  **Safety**: If Rust, mark any `unsafe` blocks and justify them. If Python, ensure no blocking I/O.
2.  **Compatibility**: Do NOT break existing `PyO3` signatures without updating the Python wrapper.
3.  **Severity**: According to `AGENTS.md`, this is a [CRITICAL/HIGH/MEDIUM/LOW] severity change.
4.  **Tests**: List which tests in `base/tests/` (Rust) or `tests/` (Python) must be run.

Provide the modified code snippet and the verification plan.