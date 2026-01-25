import {
  createContext,
  useContext,
  useState,
  useEffect,
  useCallback,
  useMemo,
  ReactNode,
} from "react";

/**
 * @module context/StreamingContext
 * @description Provides global state management for live data streaming control.
 * Acts as a "Master Switch" to gate all data connections in the application.
 * Now also respects the user's login status.
 */

/** localStorage key for persisting streaming preference */
const STORAGE_KEY = "nglab-global-streaming-enabled";

/**
 * Shape of the streaming context value.
 */
export interface StreamingContextValue {
  /** Whether global streaming is currently enabled */
  isGlobalStreamingEnabled: boolean;
  /** Toggle function to enable/disable global streaming */
  setGlobalStreamingEnabled: (enabled: boolean) => void;
  /** Whether the user is currently logged in */
  isLoggedIn: boolean;
  /** Setter for the login status */
  setIsLoggedIn: (isLoggedIn: boolean) => void;
}

/**
 * React Context for global streaming state.
 * Defaults to null; consumers must be wrapped in StreamingProvider.
 */
const StreamingContext = createContext<StreamingContextValue | null>(null);

/**
 * Props for the StreamingProvider component.
 */
interface StreamingProviderProps {
  children: ReactNode;
}

/**
 * Provider component that manages global streaming state with localStorage persistence and login gating.
 *
 * @remarks
 * - Default state is DISABLED (false) on initial load
 * - Persists to localStorage so user preference is remembered across sessions
 * - `isLoggedIn` defaults to false and should be updated by the main App component
 *
 * @example
 * ```tsx
 * <StreamingProvider>
 *   <App />
 * </StreamingProvider>
 * ```
 */
export function StreamingProvider({ children }: StreamingProviderProps) {
  // Initialize from localStorage, defaulting to false (disabled)
  const [isGlobalStreamingEnabled, setIsGlobalStreamingEnabledState] =
    useState<boolean>(() => {
      try {
        const stored = localStorage.getItem(STORAGE_KEY);
        // Default to true if nothing stored
        return stored !== null ? stored === "true" : true;
      } catch {
        // localStorage may be unavailable in some contexts
        return true;
      }
    });

  const [isLoggedIn, setIsLoggedIn] = useState<boolean>(false);

  // Persist to localStorage whenever the value changes
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, String(isGlobalStreamingEnabled));
    } catch {
      console.warn("Failed to persist streaming preference to localStorage");
    }
  }, [isGlobalStreamingEnabled]);

  // Stable setter function
  const setGlobalStreamingEnabled = useCallback((enabled: boolean) => {
    setIsGlobalStreamingEnabledState(enabled);
  }, []);

  const value: StreamingContextValue = useMemo(
    () => ({
      isGlobalStreamingEnabled,
      setGlobalStreamingEnabled,
      isLoggedIn,
      setIsLoggedIn,
    }),
    [
      isGlobalStreamingEnabled,
      setGlobalStreamingEnabled,
      isLoggedIn,
      setIsLoggedIn,
    ],
  );

  return (
    <StreamingContext.Provider value={value}>
      {children}
    </StreamingContext.Provider>
  );
}

/**
 * Hook to access global streaming state and controls.
 *
 * @throws Error if used outside of StreamingProvider
 *
 * @returns The streaming context value containing:
 * - `isGlobalStreamingEnabled`: Current state of the global streaming toggle
 * - `setGlobalStreamingEnabled`: Function to update the streaming state
 * - `isLoggedIn`: Current login status
 * - `setIsLoggedIn`: Function to update login status
 */
export function useStreaming(): StreamingContextValue {
  const context = useContext(StreamingContext);
  if (!context) {
    throw new Error("useStreaming must be used within a StreamingProvider");
  }
  return context;
}

/**
 * Export the context for testing purposes.
 */
export { StreamingContext };
