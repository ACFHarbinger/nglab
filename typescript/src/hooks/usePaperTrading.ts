import { useState, useEffect, useCallback } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";

export interface PaperAccount {
    balance: number;
    equity: number;
    positions: Record<string, number>;
    orders: any[];
}

export function usePaperTrading() {
    const [account, setAccount] = useState<PaperAccount | null>(null);
    const [isActive, setIsActive] = useState(false);

    useEffect(() => {
        // Initial fetch
        invoke<PaperAccount>("get_paper_account").then(setAccount).catch(console.error);
        invoke<boolean>("is_paper_mode_active").then(setIsActive).catch(console.error);

        // Listen for updates from simulation loop
        const unlistenPromise = listen<PaperAccount>("paper-update", (event) => {
            setAccount(event.payload);
        });

        return () => {
            unlistenPromise.then((unlisten) => unlisten());
        };
    }, []);

    const toggleMode = useCallback(async (active: boolean) => {
        try {
            await invoke("toggle_paper_mode", { active });
            setIsActive(active);
        } catch (e) {
            console.error("Failed to toggle paper mode", e);
        }
    }, []);

    const resetAccount = useCallback(async (initialBalance: number = 100000) => {
        try {
            await invoke("reset_paper_account", { initialBalance });
            const newAccount = await invoke<PaperAccount>("get_paper_account");
            setAccount(newAccount);
        } catch (e) {
            console.error("Failed to reset paper account", e);
        }
    }, []);

    return { account, isActive, toggleMode, resetAccount };
}
