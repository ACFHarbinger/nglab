import { useState, useEffect, useCallback } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";

export interface OutcomeInfo {
    id: string;
    name: string;
}

export interface MarketMetadata {
    title: string;
    outcomes: OutcomeInfo[];
}

export function usePolymarket() {
    const [livePrices, setLivePrices] = useState<Record<string, number>>({});
    const [isStreaming, setIsStreaming] = useState(false);
    const [activeMarket, setActiveMarket] = useState<MarketMetadata | null>(null);

    useEffect(() => {
        let unlisten: (() => void) | undefined;

        const setupListener = async () => {
            console.log('📡 Setting up polymarket-price-update event listener...');
            unlisten = await listen<{ asset_id: string; price: number }>('polymarket-price-update', (event) => {
                console.log('📊 Price update received:', event.payload);
                setLivePrices(prev => {
                    const updated = {
                        ...prev,
                        [event.payload.asset_id]: event.payload.price
                    };
                    console.log('📊 Updated livePrices state:', Object.keys(updated).length, 'assets');
                    return updated;
                });
            });
            console.log('✓ Event listener setup complete');
        };

        setupListener();

        return () => {
            console.log('🔌 Cleaning up polymarket event listener');
            if (unlisten) unlisten();
        };
    }, []);

    const startStream = useCallback(async (marketSource: string, metadata: MarketMetadata) => {
        if (isStreaming) {
            console.log('⚠ Stream already running, ignoring request');
            return;
        }

        console.log('🚀 Starting Polymarket stream for:', marketSource);
        console.log('🚀 Market metadata:', metadata);
        console.log('🚀 Number of outcomes:', metadata.outcomes.length);

        // Clear old prices when starting new stream
        setLivePrices({});
        setIsStreaming(true);
        setActiveMarket(metadata);

        try {
            console.log('🔄 Invoking stream_polymarket_prices command...');
            const result = await invoke('stream_polymarket_prices', { marketSource });
            console.log('✓ Stream command returned:', result);
        } catch (e) {
            console.error("❌ Failed to start stream:");
            console.error("Error details:", e);
            console.error("Error type:", typeof e);
            console.error("Error JSON:", JSON.stringify(e, null, 2));
            setIsStreaming(false);
            alert(`Failed to start stream: ${e}`);
            throw e;
        }
    }, [isStreaming]);

    const stopStream = useCallback(async () => {
        if (!isStreaming) {
            console.log('⚠ No stream running');
            return;
        }

        console.log('🛑 Stopping Polymarket stream...');
        setIsStreaming(false);

        try {
            await invoke('stop_polymarket_stream');
            console.log('✓ Stream stopped successfully');

            // Clear prices and market
            setLivePrices({});
            setActiveMarket(null);
        } catch (e) {
            console.error("❌ Failed to stop stream:", e);
            // Still update UI state even if backend call fails
            setLivePrices({});
            setActiveMarket(null);
        }
    }, [isStreaming]);

    return {
        livePrices,
        isStreaming,
        activeMarket,
        startStream,
        stopStream,
        setActiveMarket
    };
}
