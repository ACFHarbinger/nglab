import { useState, useEffect, useCallback, useRef } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";

/**
 * @module hooks/usePolymarket
 * @description Provides a React hook for interacting with Polymarket real-time price streams.
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
 * @returns {object} An object containing:
 * - `livePrices`: Record of asset IDs to their latest prices.
 * - `isStreaming`: Boolean indicating if a price stream is active.
 * - `activeMarket`: Metadata of the currently streamed market.
 * - `startStream`: Function to initiate a new price stream.
 * - `stopStream`: Function to terminate the active stream.
 * - `setActiveMarket`: Function to manually update the active market state.
 */
export function usePolymarket() {
  const [livePrices, setLivePrices] = useState<Record<string, number>>({});
  const [isStreaming, setIsStreaming] = useState(false);
  const [activeMarket, setActiveMarket] = useState<MarketMetadata | null>(null);

  // Use a ref to track streaming state for stable callbacks
  const streamingRef = useRef(false);

  useEffect(() => {
    let unlisten: (() => void) | undefined;

    const setupListener = async () => {
      console.log("📡 Setting up polymarket-price-update event listener...");
      unlisten = await listen<{ asset_id: string; price: number }>(
        "polymarket-price-update",
        (event) => {
          setLivePrices((prev) => ({
            ...prev,
            [event.payload.asset_id]: event.payload.price,
          }));
        },
      );
      console.log("✓ Event listener setup complete");
    };

    setupListener();

    return () => {
      console.log("🔌 Cleaning up polymarket event listener");
      if (unlisten) unlisten();
    };
  }, []);

  const startStream = useCallback(
    async (marketSource: string, metadata?: MarketMetadata) => {
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
        console.log("✓ Stream started with resolved metadata:", resolvedMetadata);
        setActiveMarket(resolvedMetadata);
      } catch (e) {
        console.error("❌ Failed to start stream:", e);
        setIsStreaming(false);
        streamingRef.current = false;
        throw e;
      }
    },
    [], // Stable identity
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
  };
}
