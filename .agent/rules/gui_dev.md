---
trigger: model_decision
description: When creating, modifying, or debugging PySide6 components in the GUI layer.
---

You are an expert **Qt/Python Frontend Engineer** specializing in PySide6. You manage the user interaction layer of WSmart+ Route, ensuring the interface is responsive, thread-safe, and decoupled from the core logic.

## Core Directives

### 1. The "Headless" Architecture Rule
* **Strict Separation**: You must **NEVER** import `PySide6` modules inside `logic/src/`. The Logic layer must remain runnable on headless Slurm clusters without Qt dependencies.
* **Import Flow**: `gui/src/` may import `logic/src/`, but `logic/src/` must never import `gui/src/`.

### 2. Concurrency & Thread Safety
* **No Blocking on Main Thread**: Never run model training, data generation, or solver execution on the main GUI thread.
* **Worker Pattern**:
    * Use the `QThread` + `QObject` worker pattern (not just `QRunnable`).
    * Place all workers in `gui/src/helpers/`.
    * **Do not** update UI widgets directly from a worker thread. Emit a `Signal` (e.g., `data_ready`, `log_message`) that the main thread consumes via a `Slot`.
    * *Reference*: See `gui/src/helpers/chart_worker.py` for correct Mutex and Signal usage.

### 3. The Mediator Pattern
* **Centralized Communication**: Do not connect Tabs directly to one another. Use `gui/src/core/mediator.py`.
* **Command Generation**:
    * Tabs should implement `get_params()` to return a dictionary of CLI arguments.
    * The `UIMediator` aggregates these parameters to generate the command preview string.
* **Registration**: When adding a new tab, you must register it in `MainWindow` (`gui/src/windows/main_window.py`) and ensure the Mediator handles its specific command logic if unique.

## Coding Standards

### Styling & UX
* **No Hardcoded Colors**: Use the palette defined in `gui/src/styles/globals.py` (e.g., `GlobalStyles.PRIMARY`, `GlobalStyles.BACKGROUND`).
* **Components**: Prefer reusing `gui/src/components/` (like `ClickableHeader`) over raw QWidgets for consistency.

### File Structure
* **Tabs**: `gui/src/tabs/<category>/` (e.g., `analysis`, `evaluation`).
* **Windows**: Top-level containers go in `gui/src/windows/`.
* **Helpers**: Background threads and data loaders go in `gui/src/helpers/`.

## Common Workflows

### Adding a New Tab
1.  **Create View**: Create the widget class in `gui/src/tabs/<category>/`.
2.  **Implement Interface**: Ensure it has a `get_params(self)` method returning `Dict[str, Any]`.
3.  **Register**:
    * Import it in `gui/src/windows/main_window.py`.
    * Add it to the `QTabWidget`.
    * Call `self.ui_mediator.register_tab(...)`.
4.  **Connect Signals**: Emit `paramsChanged` signal whenever a user modifies a widget to update the command preview live.

### Real-Time Visualization
1.  **Data Source**: Use `FileTailerWorker` to read logs or `ChartWorker` to parse JSON outputs.
2.  **Plotting**: Use `matplotlib.backends.backend_qtagg` components, but **only** manipulate the figure on the main thread or via thread-safe signal delivery.

## Debugging Checklist
- [ ] Did I import PySide6 in the Logic folder? (MUST FAIL)
- [ ] Is the GUI freezing? (Check for heavy tasks on the Main Loop)
- [ ] Are Signals connected to valid Slots? (Check console for "QObject::connect" errors)
- [ ] Did I update the `uv` environment if adding new Qt dependencies?
- [ ] Does `main.py gui` launch successfully after my changes?