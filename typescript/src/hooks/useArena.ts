import { useState, useEffect, useCallback, useRef } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { useStreamingGuard } from "./useStreamingGuard";

/**
 * @module hooks/useArena
 * @description Provides a React hook for managing the real-time trading arena simulation.
 * Respects the global streaming control - simulation will not start if streaming is disabled.
 */

/**
 * A single order in the book.
 */
export interface Order {
  id: number;
  price: number;
  quantity: number;
  filled: number;
  side: "Bid" | "Ask";
  order_type: "Limit" | "Market";
  timestamp: number;
}

/**
 * A price level in the book, containing multiple orders.
 */
export interface PriceLevel {
  price: number;
  orders: Order[];
  total_quantity: number;
}

/**
 * Snapshot of the current state of the order book.
 */
export interface OrderBook {
  /** Bids ordered by price (highest first) */
  bids: Record<string, PriceLevel>;
  /** Asks ordered by price (lowest first) */
  asks: Record<string, PriceLevel>;
  /** Server-side timestamp */
  timestamp: number;
}

/**
 * Real-time update emitted by the Tauri backend.
 */
export interface ArenaUpdate {
  /** Current simulation step */
  step: number;
  /** Current mid-price */
  price: number;
  /** Total account value in USDC */
  portfolio_value: number;
  /** Current real-time risk score (0-100) */
  risk_score: number;
  /** Current peak-to-trough drawdown */
  current_drawdown: number;
  /** Value at Risk (VaR) */
  current_var: number;
  /** Current orderbook state */
  orderbook: OrderBook;
  /** Active algorithmic execution orders */
  algo_orders: any[];
  /** Current position */
  position: number;
  /** Whether market maker mode is active */
  mm_active: boolean;
  /** Realized PnL from market making */
  mm_realized_pnl: number;
  /** Cumulative P&L */
  pnl: number;
  /** Return percentage */
  return_pct: number;
  /** Maximum drawdown */
  max_drawdown: number;
  /** Rolling volatility */
  volatility: number;
}

/**
 * Custom hook to manage real-time arena state and simulation controls.
 *
 * Listens for 'arena-update' events from the Tauri backend and maintains
 * state for the latest update, tick history, and simulation status.
 *
 * **Streaming Guard**: This hook respects the global streaming control.
 * If global streaming is disabled, `start()` will be a no-op and any
 * running simulation will be automatically stopped.
 *
 * @returns {object} An object containing:
 * - `data`: The latest `ArenaUpdate` or `null`.
 * - `history`: An array of the last 200 `ArenaUpdate` snapshots.
 * - `isRunning`: Boolean indicating if if the simulation is active.
 * - `start`: Function to start the simulation.
 * - `stop`: Function to stop the simulation.
 * - `isGlobalStreamingEnabled`: Whether global streaming is enabled.
 */
export function useArena() {
  const [data, setData] = useState<ArenaUpdate | null>(null);
  const [history, setHistory] = useState<ArenaUpdate[]>([]);
  const [isRunning, setIsRunning] = useState(false);

  // --- STREAMING GUARD: Get global streaming state ---
  const { canStream, isGlobalStreamingEnabled } = useStreamingGuard();

  // Track running state for cleanup effects
  const isRunningRef = useRef(false);
  isRunningRef.current = isRunning;

  // --- STREAMING GUARD: Auto-stop when global streaming is disabled or logged out ---
  useEffect(() => {
    if (!canStream && isRunningRef.current) {
      console.log("🛑 Streaming gated - stopping arena simulation");
      invoke("stop_simulation")
        .then(() => setIsRunning(false))
        .catch(console.error);
    }
  }, [canStream]);

  useEffect(() => {
    // --- STREAMING GUARD: Don't set up listener if streaming is gated ---
    if (!canStream) {
      return;
    }

    const unlistenPromise = listen<ArenaUpdate>("arena-update", (event) => {
      const update = event.payload;
      setData(update);
      setHistory((prev) => {
        const newHistory = [...prev, update];
        // Limit history to last 200 points to keep chart performant
        if (newHistory.length > 200) return newHistory.slice(-200);
        return newHistory;
      });
    });

    return () => {
      unlistenPromise.then((unlisten) => unlisten());
    };
  }, [canStream]);

  /**
   * Starts the simulation on the backend.
   * Sets `isRunning` to true upon successful invocation.
   *
   * **Note**: This is a no-op if global streaming is disabled.
   */
  const start = useCallback(() => {
    // --- STREAMING GUARD: Prevent start if streaming is disabled ---
    if (!canStream) {
      console.warn(
        "⚠️ Cannot start arena simulation: streaming is gated (check toggle or login)",
      );
      return;
    }

    invoke("start_simulation")
      .then(() => setIsRunning(true))
      .catch(console.error);
  }, [canStream]);

  /**
   * Stops the simulation on the backend.
   * Sets `isRunning` to false upon successful invocation.
   */
  const stop = useCallback(() => {
    invoke("stop_simulation")
      .then(() => setIsRunning(false))
      .catch(console.error);
  }, []);

  return { data, history, isRunning, start, stop, isGlobalStreamingEnabled };
}
