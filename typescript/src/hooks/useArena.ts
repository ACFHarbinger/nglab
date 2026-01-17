import { useState, useEffect, useCallback } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";

/**
 * @module hooks/useArena
 * @description Provides a React hook for managing the real-time trading arena simulation.
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
    /** Current orderbook state */
    orderbook: OrderBook;
}

/**
 * Custom hook to manage real-time arena state and simulation controls.
 *
 * Listens for 'arena-update' events from the Tauri backend and maintains
 * state for the latest update, tick history, and simulation status.
 *
 * @returns {object} An object containing:
 * - `data`: The latest `ArenaUpdate` or `null`.
 * - `history`: An array of the last 200 `ArenaUpdate` snapshots.
 * - `isRunning`: Boolean indicating if if the simulation is active.
 * - `start`: Function to start the simulation.
 * - `stop`: Function to stop the simulation.
 */
export function useArena() {
    const [data, setData] = useState<ArenaUpdate | null>(null);
    const [history, setHistory] = useState<ArenaUpdate[]>([]);
    const [isRunning, setIsRunning] = useState(false);

    useEffect(() => {
        const unlistenPromise = listen<ArenaUpdate>("arena-update", (event) => {
            const update = event.payload;
            setData(update);
            setHistory(prev => {
                const newHistory = [...prev, update];
                // Limit history to last 200 points to keep chart performant
                if (newHistory.length > 200) return newHistory.slice(-200);
                return newHistory;
            });
        });

        return () => {
            unlistenPromise.then(unlisten => unlisten());
        };
    }, []);

    /**
     * Starts the simulation on the backend.
     * Sets `isRunning` to true upon successful invocation.
     */
    const start = useCallback(() => {
        invoke("start_simulation").then(() => setIsRunning(true)).catch(console.error);
    }, []);

    /**
     * Stops the simulation on the backend.
     * Sets `isRunning` to false upon successful invocation.
     */
    const stop = useCallback(() => {
        invoke("stop_simulation").then(() => setIsRunning(false)).catch(console.error);
    }, []);

    return { data, history, isRunning, start, stop };
}
