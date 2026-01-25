import { useContext } from "react";
import { StreamingContext } from "../context/StreamingContext";

/**
 * @module hooks/useStreamingGuard
 * @description Provides a reusable guard pattern for data streaming hooks.
 * Use this in any data source hook (Kalshi, Yahoo Finance, etc.) to respect
 * the global streaming control and user login status.
 */

/**
 * Return type for the useStreamingGuard hook.
 */
export interface StreamingGuardResult {
  /** Whether streaming operations can proceed (global streaming enabled AND user logged in) */
  canStream: boolean;
  /** Current state of the global streaming toggle */
  isGlobalStreamingEnabled: boolean;
  /** Current login status */
  isLoggedIn: boolean;
}

/**
 * A reusable guard hook that checks whether streaming is allowed.
 *
 * Use this hook in any data streaming hook to gate operations based on
 * the global streaming control and login status.
 *
 * @returns Object containing:
 * - `canStream`: Boolean indicating if streaming operations should proceed
 * - `isGlobalStreamingEnabled`: Current state of the global toggle
 * - `isLoggedIn`: Current login status
 *
 * @example
 * ```tsx
 * function useKalshiMarkets() {
 *   const { canStream, isLoggedIn } = useStreamingGuard();
 *
 *   const startStream = useCallback(async () => {
 *     // --- STREAMING GUARD: Prevent streaming if global toggle is off or user not logged in ---
 *     if (!canStream) {
 *       const reason = !isLoggedIn ? "User not logged in" : "Global streaming is disabled";
 *       console.warn(`Streaming is gated: ${reason}`);
 *       return;
 *     }
 *     // ... streaming logic
 *   }, [canStream, isLoggedIn]);
 * }
 * ```
 */
export function useStreamingGuard(): StreamingGuardResult {
  // We access specific context directly here to avoid throwing if the provider is missing
  // This makes the widget portable (e.g. tests, storybook) without crashing
  const context = useContext(StreamingContext);

  // If context is missing, default to safe values (no streaming, not logged in)
  if (!context) {
    if (process.env.NODE_ENV === "development") {
      console.warn(
        "useStreamingGuard: StreamingContext is missing. Defaulting to disabled/logged out.",
      );
    }
    return {
      canStream: false,
      isGlobalStreamingEnabled: false,
      isLoggedIn: false,
    };
  }

  const { isGlobalStreamingEnabled, isLoggedIn } = context;

  return {
    canStream: isGlobalStreamingEnabled && isLoggedIn,
    isGlobalStreamingEnabled,
    isLoggedIn,
  };
}
