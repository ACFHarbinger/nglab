# ADR-0004: Use Tauri for Frontend

## Status
Accepted

## Context
NGLab requires a graphical user interface (GUI) to visualize simulation data, such as real-time order book charts, portfolio performance, and training metrics. The application needs to be cross-platform, efficient, and tightly integrated with the high-performance Rust backend.

## Decision
We will use **Tauri (v2)** to build the frontend application.
- **Backend**: Rust (using the same codebase as the simulation engine).
- **Frontend**: React + TypeScript + Tailwind CSS.
- **Communication**: Tauri's IPC mechanism (events and commands) will stream updates from the `Arena` to the UI.

## Consequences
- **Easier**:
    - **Performance**: Drastically smaller application size and memory footprint compared to Electron.
    - **Security**: Rust backend provides memory safety; Tauri's security model isolates the frontend from system APIs.
    - **Integration**: Direct access to the Rust simulation structs (e.g., `OrderBook`) allows for efficient data transformation before sending to the UI.
- **Difficult**:
    - **Browser Compatibility**: Relies on the host operating system's WebView (WebKit on Linux/macOS, WebView2 on Windows), which can lead to minor rendering differences.
    - **Mobile**: While Tauri 2.0 supports mobile, it is less mature than React Native or Flutter.

## Alternatives Considered
- **Electron**: The industry standard, but notoriously resource-heavy (bundled Chromium). We prioritized performance and binary size.
- **Native GUI (Iced/Egui)**: Would offer the best theoretical performance but lacks the rich ecosystem of web visualization libraries (e.g., TradingView charts) that we need for financial data.
- **Web App Only**: Hosting a local web server (e.g., Python `FastAPI`). This adds latency and doesn't provide the native desktop experience (system tray, native menus, file system access) we desire.
