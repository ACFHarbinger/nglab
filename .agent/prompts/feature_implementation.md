# Feature Implementation Prompt

**Intent:** Implement a new feature while maintaining architectural boundaries in Image-Toolkit.

## The Prompt

I need to implement a new feature: `[INSERT FEATURE NAME]`.
**Goal:** [Brief description of what the feature does].
**Dependencies:** [List relevant tabs, backend modules, or Rust functions].

**Strict Constraints:**
1. **Separation**: If it involves heavy pixel logic -> `base/`. If it's UI -> `gui/` or `frontend/`.
2. **Concurrency**: All long-running tasks must go into `gui/src/helpers/`.
3. **Database**: Updates to schema must use `pgvector` compatible migrations.
4. **Security**: No hardcoded secrets. Use `VaultManager`.

Provide a plan or code snippet for the implementation, ensuring the UI remains responsive.
