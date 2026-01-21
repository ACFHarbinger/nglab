# ADR-0010: Use React Hooks for State Management

## Status
Accepted

## Context
NGLab is a "local-first" application where the primary source of truth is often the Rust backend (e.g., the `Arena` state). The frontend acts mainly as a view layer. Importing heavy state management libraries like Redux adds boilerplate and complexity that may not be justified given that we stream data updates directly to components.

## Decision
We will rely on **React Hooks** and **Context API** for frontend state management.
- **Local State**: `useState` and `useReducer` for component-specific logic (e.g., form inputs, toggle states).
- **Shared State**: React `Context` for globally accessible data (e.g., `ArenaContext`, `ThemeContext`).
- **Data Fetching**: Custom hooks (e.g., `useArena`, `usePolymarket`) will encapsulate data fetching and subscription logic.

## Consequences
- **Easier**:
    - **Simplicity**: No need to define actions, reducers, thunks, or sagas for simple updates.
    - **Dependency**: Zero additional dependencies to maintain or upgrade.
    - **Performance**: We can optimize re-renders by splitting Contexts granularly.
- **Difficult**:
    - **Prop Drilling**: Passing data deep down the tree can be verbose (solved by Context).
    - **Debugging**: Inspecting complex state changes is harder than with Redux DevTools (though React DevTools helps).

## Alternatives Considered
- **Redux (Toolkit)**: Powerful but excessive boilerplate for our needs.
- **Zustand**: A lightweight alternative we might consider if Context performance becomes an issue, but for now, vanilla React is sufficient.
- **Recoil/Jotai**: Atomic state management is great for complex inter-dependencies, but our data flow is mostly unidirectional (Backend -> Frontend).
