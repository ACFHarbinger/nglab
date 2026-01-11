# Documentation Update Prompt

**Intent:** Ensure documentation remains accurate after code changes.

## The Prompt

I have modified the following files: `[INSERT FILE LIST]`.

Please update the relevant documentation to reflect these changes:
1. **`AGENTS.md`**: If architectural boundaries or governances changed.
2. **`README.md`**: If new dependencies or setup steps are needed.
3. **`task.md`**: Mark relevant items as complete.
4. **Docstrings**: Ensure all new functions have Google-style docstrings.

Verify if any new `Cargo.toml` dependencies need to be listed or if `npm install` is required.