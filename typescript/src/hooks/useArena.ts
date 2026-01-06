import { useState, useEffect, useCallback } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";

export interface Order {
    id: number;
    price: number;
    quantity: number;
    filled: number;
    side: "Bid" | "Ask";
    order_type: "Limit" | "Market";
    timestamp: number;
}

export interface PriceLevel {
    price: number;
    orders: Order[];
    total_quantity: number;
}

// IndexMap<i64, PriceLevel> serializes to JSON object with string integer keys
export interface OrderBook {
    bids: Record<string, PriceLevel>;
    asks: Record<string, PriceLevel>;
    timestamp: number;
}

export interface ArenaUpdate {
    step: number;
    price: number;
    portfolio_value: number;
    orderbook: OrderBook;
}

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

    const start = useCallback(() => {
        invoke("start_simulation").then(() => setIsRunning(true)).catch(console.error);
    }, []);

    const stop = useCallback(() => {
        invoke("stop_simulation").then(() => setIsRunning(false)).catch(console.error);
    }, []);

    return { data, history, isRunning, start, stop };
}
