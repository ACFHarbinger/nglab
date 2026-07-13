import { useState, useEffect, useCallback, useRef } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { useStreaming, StreamingStats } from "../context/StreamingContext";

/**
 * @module hooks/usePolymarket
 * @description Provides a React hook for interacting with Polymarket real-time price streams.
 * Respects the global streaming control - streams will not start if streaming is disabled.
 */

export interface OutcomeInfo {
  id: string;
  name: string;
}

/**
 * Metadata for a Polymarket market.
 */
export interface MarketMetadata {
  /** The title or question of the market. */
  title: string;
  /** List of possible outcomes and their IDs. */
  outcomes: OutcomeInfo[];
}

/**
 * Custom hook to manage Polymarket live price streaming and metadata.
 *
 * Listens for 'polymarket-price-update' events from the Tauri backend
 * and maintains the current live prices and active market state.
 *
 * **Streaming Guard**: This hook respects the global streaming control.
 * If global streaming is disabled, `startStream()` will throw an error
 * and any active stream will be automatically disconnected.
 *
 * @returns {object} An object containing:
 * - `livePrices`: Record of asset IDs to their latest prices.
 * - `isStreaming`: Boolean indicating if a price stream is active.
 * - `activeMarket`: Metadata of the currently streamed market.
 * - `startStream`: Function to initiate a new price stream.
 * - `stopStream`: Function to terminate the active stream.
 * - `setActiveMarket`: Function to manually update the active market state.
 * - `isGlobalStreamingEnabled`: Whether global streaming is enabled.
 */
export function usePolymarket() {
  const [livePrices, setLivePrices] = useState<Record<string, number>>({});
  const [isStreaming, setIsStreaming] = useState(false);
  const [activeMarket, setActiveMarket] = useState<MarketMetadata | null>(null);

  // --- STREAMING GUARD: Get global streaming state and stats setter ---
  const { isGlobalStreamingEnabled, isLoggedIn, setStats } = useStreaming();
  const canStream = isGlobalStreamingEnabled && isLoggedIn;

  // Use a ref to track streaming state for stable callbacks
  const streamingRef = useRef(false);

  // --- STREAMING GUARD: Auto-disconnect when streaming is gated (toggle off or logged out) ---
  useEffect(() => {
    if (!canStream && streamingRef.current) {
      console.log("🛑 Streaming gated - stopping Polymarket stream");
      setIsStreaming(false);
      streamingRef.current = false;

      invoke("stop_polymarket_stream")
        .then(() => {
          setLivePrices({});
          setActiveMarket(null);
          setStats({
            latencyMs: 0,
            msgsPerSec: 0,
            status: "idle",
            statusMessage: "Stream stopped",
          });
        })
        .catch(console.error);
    }
  }, [canStream, setStats]);

  useEffect(() => {
    // --- STREAMING GUARD: Don't set up listener if streaming is gated ---
    if (!canStream) {
      return;
    }

    let unlisten: (() => void) | undefined;

    const setupListener = async () => {
      console.log("📡 Setting up polymarket-streaming-event listener...");
      unlisten = await listen<any>("polymarket-streaming-event", (event) => {
        const payload = event.payload;

        if (payload.type === "data") {
          setLivePrices((prev: Record<string, number>) => ({
            ...prev,
            [payload.asset_id]: payload.price,
          }));
        } else if (payload.type === "health") {
          setStats((prev: StreamingStats) => ({
            ...prev,
            latencyMs: payload.latency_ms,
            msgsPerSec: payload.msgs_per_sec,
          }));
        } else if (payload.type === "status") {
          setStats((prev: StreamingStats) => ({
            ...prev,
            status: payload.status as any,
            statusMessage: payload.message,
          }));
        }
      });
      console.log("✓ Event listener setup complete");
    };

    setupListener();

    return () => {
      console.log("🔌 Cleaning up polymarket event listener");
      if (unlisten) unlisten();
    };
  }, [canStream, setStats]);

  const startStream = useCallback(
    async (marketSource: string, metadata?: MarketMetadata) => {
      // --- STREAMING GUARD: Prevent start if streaming is disabled ---
      if (!canStream) {
        console.warn(
          "⚠️ Cannot start Polymarket stream: streaming is gated (check toggle or login)",
        );
        throw new Error(
          "Streaming is gated. Ensure you are logged in and master switch is ON.",
        );
      }

      console.log("🚀 Starting Polymarket stream for:", marketSource);

      // Clear old prices and update state
      setLivePrices({});
      setIsStreaming(true);
      streamingRef.current = true;

      if (metadata) {
        setActiveMarket(metadata);
      }

      try {
        // Now returns actual metadata from the backend
        const resolvedMetadata: MarketMetadata = await invoke(
          "stream_polymarket_prices",
          { marketSource },
        );
        console.log(
          "✓ Stream started with resolved metadata:",
          resolvedMetadata,
        );
        setActiveMarket(resolvedMetadata);
      } catch (e) {
        console.error("❌ Failed to start stream:", e);
        setIsStreaming(false);
        streamingRef.current = false;
        throw e;
      }
    },
    [canStream],
  );

  const stopStream = useCallback(async () => {
    if (!streamingRef.current) return;

    console.log("🛑 Stopping Polymarket stream...");
    setIsStreaming(false);
    streamingRef.current = false;

    try {
      await invoke("stop_polymarket_stream");
      setLivePrices({});
      setActiveMarket(null);
    } catch (e) {
      console.error("❌ Failed to stop stream:", e);
      setLivePrices({});
      setActiveMarket(null);
    }
  }, []); // Stable identity

  return {
    livePrices,
    isStreaming,
    activeMarket,
    startStream,
    stopStream,
    setActiveMarket,
    isGlobalStreamingEnabled,
  };
}
