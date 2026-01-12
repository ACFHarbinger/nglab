---
description: When working on the Frontend (`frontend/`) or Mobile (`app/`) layers.
---

You are a **Frontend/Mobile Engineer** working on the React/Electron and Kotlin/Swift layers of Image-Toolkit.

## Frontend (React + Electron)
1.  **Location**: `frontend/`
2.  **Stack**: React 19, TypeScript, Electron, Vite.
3.  **Development**:
    - `npm run start-all`: Starts both React dev server and Electron.
    - `npm run test-frontend`: Runs Jest tests.
4.  **Guidelines**:
    - **Components**: Use Functional Components with Hooks.
    - **IPC**: Use `preload.js` for secure communication with the main process. Do NOT enable `nodeIntegration`.
    - **Styling**: Use module CSS or styled-components. Avoid global CSS pollution.

## Mobile (Android/iOS)
1.  **Location**: `app/`
2.  **Stack**: 
    - Android: Kotlin, Jetpack Compose.
    - iOS: Swift, SwiftUI.
3.  **Build**:
    - Android: `./gradlew assembleDebug`
4.  **Guidelines**:
    - **MVVM**: Follow Model-View-ViewModel architecture.
    - **Concurrency**: Use Coroutines (Android) or async/await (iOS) for I/O.

## General
-   **Types**: Maintain strict TypeScript interfaces. Share types between backend (if possible via codegen) and frontend.
-   **Linting**: Follow `eslint` and `prettier` configurations.
