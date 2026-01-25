import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import { StreamingProvider } from "./context/StreamingContext";
import "./index.css";

import { ErrorBoundary } from "./components/ErrorBoundary";

/**
 * Entry point for the NGLab React application.
 *
 * Mounts the root App component to the DOM in StrictMode.
 * StreamingProvider wraps the App to provide global streaming control.
 */
ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <ErrorBoundary>
      <StreamingProvider>
        <App />
      </StreamingProvider>
    </ErrorBoundary>
  </React.StrictMode>,
);
