import { useState, useEffect, useCallback } from "react";
import { invoke } from "@tauri-apps/api/core";

/**
 * Hook to manage active exchange state and listing.
 */
export function useExchange() {
    const [exchanges, setExchanges] = useState<string[]>([]);
    const [activeExchange, setActiveExchangeState] = useState<string>("");
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const init = async () => {
            try {
                const list: string[] = await invoke("list_exchanges");
                const active: string = await invoke("get_active_exchange");
                setExchanges(list);
                setActiveExchangeState(active);
            } catch (e) {
                console.error("Failed to initialize exchanges:", e);
            } finally {
                setLoading(false);
            }
        };
        init();
    }, []);

    const setActiveExchange = useCallback(async (name: string) => {
        try {
            await invoke("set_active_exchange", { name });
            setActiveExchangeState(name);
        } catch (e) {
            console.error("Failed to set active exchange:", e);
            throw e;
        }
    }, []);

    return {
        exchanges,
        activeExchange,
        setActiveExchange,
        loading,
    };
}
