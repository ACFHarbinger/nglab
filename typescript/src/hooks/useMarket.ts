import { useState, useCallback, useEffect } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";

export interface MarketSearchResult {
    id: string;
    title: string;
    description?: string;
    outcomes: string[];
    token_ids: string[];
    volume?: number;
    liquidity?: number;
    active: boolean;
}

export interface MarketData {
    symbol: string;
    price: number;
    volume: number;
    best_bid: number;
    best_ask: number;
}

export interface PriceUpdate {
    exchange: string;
    symbol: string;
    price: number;
}

/**
 * Hook for searching and interacting with markets on the active exchange.
 */
export function useMarket() {
    const [searchResults, setSearchResults] = useState<MarketSearchResult[]>([]);
    const [isSearching, setIsSearching] = useState(false);
    const [livePrices, setLivePrices] = useState<Record<string, number>>({});

    useEffect(() => {
        let unlisten: (() => void) | undefined;
        const setupListener = async () => {
            unlisten = await listen<PriceUpdate>("price-update", (event) => {
                setLivePrices((prev) => ({
                    ...prev,
                    [event.payload.symbol]: event.payload.price,
                }));
            });
        };
        setupListener();
        return () => { if (unlisten) unlisten(); };
    }, []);

    /**
     * Search for markets on the active exchange.
     */
    const searchMarkets = useCallback(async (query: string, limit: number = 20) => {
        if (!query) {
            setSearchResults([]);
            return;
        }
        setIsSearching(true);
        try {
            const results: MarketSearchResult[] = await invoke("search_exchange_markets", { query, limit });
            setSearchResults(results || []);
        } catch (e) {
            console.error("Market search failed:", e);
            setSearchResults([]);
        } finally {
            setIsSearching(false);
        }
    }, []);

    /**
     * Start streaming prices for a list of symbols.
     */
    const startStream = useCallback(async (symbols: string[]) => {
        try {
            await invoke("stream_exchange_prices", { symbols });
        } catch (e) {
            console.error("Failed to start price stream:", e);
        }
    }, []);

    /**
     * Get current market data for a specific symbol.
     */
    const getMarketData = useCallback(async (symbol: string): Promise<MarketData | null> => {
        try {
            return await invoke("get_exchange_market_data", { symbol });
        } catch (e) {
            console.error("Failed to fetch market data:", e);
            return null;
        }
    }, []);

    return {
        searchResults,
        isSearching,
        livePrices,
        searchMarkets,
        startStream,
        getMarketData,
    };
}
