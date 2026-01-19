import React, { useState, useEffect } from "react";
import { invoke } from "@tauri-apps/api/core";
import { MarketMetadata } from "../hooks/usePolymarket";
import {
    User,
    Shield,
    Plus,
    Trash2,
    Settings,
    AlertCircle,
    Globe,
    ExternalLink,
    RefreshCw,
    Lock,
    Bug
} from "lucide-react";
import clsx from "clsx";

interface IntegrationConfig {
    service: string;
    config: any;
}

interface ExternalIntegration {
    id: number;
    service_name: string;
    config: IntegrationConfig;
    created_at: string;
}

interface Props {
    isStreaming?: boolean;
    startStream?: (marketSource: string, metadata: MarketMetadata) => Promise<void>;
    stopStream?: () => Promise<void>;
}

/**
 * AccountTab component for managing user profile and external integrations.
 */
export function AccountTab({ isStreaming = false, startStream, stopStream }: Props) {
    const [integrations, setIntegrations] = useState<ExternalIntegration[]>([]);
    const [isVaultUnlocked, setIsVaultUnlocked] = useState(false);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [debugMode, setDebugMode] = useState(false);

    // Form states for Polymarket
    const [apiKey, setApiKey] = useState("");
    const [secret, setSecret] = useState("");
    const [passphrase, setPassphrase] = useState("");
    const [proxyAddress, setProxyAddress] = useState("");

    useEffect(() => {
        checkVaultStatus();
    }, []);

    const checkVaultStatus = async () => {
        try {
            const unlocked = await invoke<boolean>("is_vault_unlocked");
            setIsVaultUnlocked(unlocked);
            if (unlocked) {
                fetchIntegrations();
            }
        } catch (err) {
            console.error("Failed to check vault status", err);
        }
    };

    const fetchIntegrations = async () => {
        setIsLoading(true);
        try {
            const response: any = await invoke("list_integrations");
            if (response.success) {
                setIntegrations(response.data || []);
            } else {
                setError(response.message);
            }
        } catch (err) {
            setError("Failed to fetch integrations");
        } finally {
            setIsLoading(false);
        }
    };

    const handleSavePolymarket = async (e: React.FormEvent) => {
        e.preventDefault();
        setIsLoading(true);
        setError(null);
        try {
            const response: any = await invoke("save_polymarket_integration", {
                apiKey,
                secret,
                passphrase,
                proxyAddress: proxyAddress || null
            });
            if (response.success) {
                setApiKey("");
                setSecret("");
                setPassphrase("");
                setProxyAddress("");
                fetchIntegrations();
            } else {
                setError(response.message);
            }
        } catch (err) {
            setError("Failed to save Polymarket integration");
        } finally {
            setIsLoading(false);
        }
    };

    const handleDeleteIntegration = async (id: number) => {
        if (!confirm("Are you sure you want to remove this integration?")) return;
        try {
            const response: any = await invoke("delete_integration", { id });
            if (response.success) {
                setIntegrations(prev => prev.filter(i => i.id !== id));
            } else {
                setError(response.message);
            }
        } catch (err) {
            setError("Failed to delete integration");
        }
    };

    const toggleDebugMode = async () => {
        const nextMode = !debugMode;
        setDebugMode(nextMode);
        try {
            await invoke("set_debug_mode", { enabled: nextMode });
        } catch (err) {
            console.error("Failed to set debug mode", err);
        }
    };

    if (!isVaultUnlocked) {
        return (
            <div className="flex-1 flex flex-col items-center justify-center p-6 text-center">
                <div className="w-16 h-16 bg-slate-900 rounded-2xl flex items-center justify-center mb-4 border border-slate-800 shadow-xl">
                    <Lock className="w-8 h-8 text-indigo-400 opacity-50" />
                </div>
                <h3 className="text-xl font-bold text-white mb-2">Vault Locked</h3>
                <p className="text-slate-400 max-w-sm mb-6">
                    External integrations are securely stored in your encrypted vault. Please unlock your vault to manage connections.
                </p>
            </div>
        );
    }

    return (
        <div className="flex-1 flex flex-col p-6 gap-8 overflow-y-auto custom-scrollbar">
            {/* Account Header */}
            <div className="flex items-center justify-between">
                <div className="flex items-center gap-4">
                    <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center shadow-lg shadow-indigo-500/20">
                        <User className="w-7 h-7 text-white" />
                    </div>
                    <div>
                        <h2 className="text-2xl font-bold text-white tracking-tight">Account Settings</h2>
                        <p className="text-slate-400 text-sm flex items-center gap-1.5 mt-0.5">
                            <Shield className="w-3.5 h-3.5 text-emerald-400" />
                            Vault Security Active
                        </p>
                    </div>
                </div>

                <div className="flex items-center gap-3">
                    <button
                        onClick={fetchIntegrations}
                        className="p-2.5 rounded-xl bg-slate-900 border border-slate-800 text-slate-400 hover:text-white transition-all shadow-sm"
                    >
                        <RefreshCw className={clsx("w-5 h-5", isLoading && "animate-spin")} />
                    </button>
                </div>
            </div>

            <div className="grid grid-cols-1 xl:grid-cols-3 gap-8">
                {/* Left: Active Integrations */}
                <div className="xl:col-span-2 space-y-6">
                    <div className="flex items-center justify-between">
                        <h3 className="text-sm font-semibold uppercase tracking-widest text-slate-500">Active Connections</h3>
                        <span className="text-xs text-slate-600 bg-slate-900 px-2 py-0.5 rounded-md border border-slate-800 font-mono">
                            {integrations.length} Integrations
                        </span>
                    </div>

                    {integrations.length === 0 ? (
                        <div className="py-12 flex flex-col items-center justify-center bg-slate-900/30 border border-dashed border-slate-800 rounded-3xl">
                            <Globe className="w-12 h-12 text-slate-700 mb-4" />
                            <p className="text-slate-500 font-medium">No external accounts connected yet.</p>
                            <p className="text-slate-600 text-sm mt-1">Add your first integration on the right.</p>
                        </div>
                    ) : (
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            {integrations.map((integration) => (
                                <div
                                    key={integration.id}
                                    className="group relative bg-slate-900 border border-slate-800 p-5 rounded-2xl hover:border-slate-700 transition-all shadow-lg hover:shadow-indigo-500/5"
                                >
                                    <div className="flex items-start justify-between mb-4">
                                        <div className="flex items-center gap-3">
                                            <div className="w-10 h-10 rounded-xl bg-indigo-500/10 flex items-center justify-center border border-indigo-500/20">
                                                <Globe className="w-5 h-5 text-indigo-400" />
                                            </div>
                                            <div>
                                                <h4 className="font-bold text-white capitalize">{integration.service_name}</h4>
                                                <div className="flex items-center gap-1.5 mt-0.5">
                                                    <div className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
                                                    <span className="text-[10px] text-emerald-400 font-bold uppercase tracking-wider">Connected</span>
                                                </div>
                                            </div>
                                        </div>
                                        <button
                                            onClick={() => handleDeleteIntegration(integration.id)}
                                            className="p-2 text-slate-600 hover:text-rose-400 hover:bg-rose-400/10 rounded-lg transition-all opacity-0 group-hover:opacity-100"
                                        >
                                            <Trash2 size={16} />
                                        </button>
                                    </div>

                                    <div className="space-y-2 mb-4">
                                        <div className="flex flex-col">
                                            <span className="text-[10px] text-slate-500 uppercase tracking-wider font-bold mb-1">Created At</span>
                                            <span className="text-xs text-slate-300 font-mono">{new Date(integration.created_at).toLocaleDateString()}</span>
                                        </div>
                                    </div>

                                    <div className="mt-4 pt-4 border-t border-slate-800/50 flex items-center justify-between">
                                        <div className="flex items-center gap-2">
                                            <div className={`w-2 h-2 rounded-full ${isStreaming ? "bg-green-500 animate-pulse" : "bg-slate-500"}`} />
                                            <span className="text-xs text-slate-400 font-medium">
                                                {isStreaming ? "Live Data Active" : "Stream Idle"}
                                            </span>
                                        </div>
                                        <div className="flex gap-2">
                                            {!isStreaming && (
                                                <button
                                                    onClick={() => startStream && startStream("will-trump-nominate-kevin-warsh-as-the-next-fed-chair", {
                                                        title: "Kevin Warsh for Fed Chair",
                                                        outcomes: [
                                                            { id: "51338236787729560681434534660841415073585974762690814047670810862722808070955", name: "Yes" },
                                                            { id: "18289842382539867639079362738467334752951741961393928566628307174343542320349", name: "No" }
                                                        ]
                                                    })}
                                                    className="text-[10px] font-bold text-emerald-400 hover:text-emerald-300 border border-emerald-500/30 hover:bg-emerald-500/10 px-2 py-1 rounded transition-colors"
                                                >
                                                    TEST STREAM
                                                </button>
                                            )}
                                            {isStreaming && (
                                                <button
                                                    onClick={() => stopStream && stopStream()}
                                                    className="text-[10px] font-bold text-rose-400 hover:text-rose-300 border border-rose-500/30 hover:bg-rose-500/10 px-2 py-1 rounded transition-colors"
                                                >
                                                    STOP STREAM
                                                </button>
                                            )}
                                        </div>
                                    </div>

                                    <div className="flex items-center justify-between pt-4 border-t border-slate-800/50">
                                        <span className="text-[10px] text-slate-600 font-mono uppercase">ID: {integration.id}</span>
                                        <a href="#" className="flex items-center gap-1.5 text-indigo-400 hover:text-indigo-300 text-xs font-semibold transition-colors">
                                            Service Status <ExternalLink size={12} />
                                        </a>
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}
                </div>

                {/* Right: Add Form */}
                <div className="space-y-6">
                    <h3 className="text-sm font-semibold uppercase tracking-widest text-slate-500">New Integration</h3>

                    <div className="bg-slate-900 border border-slate-800 p-6 rounded-3xl shadow-2xl relative overflow-hidden group">
                        <div className="absolute top-0 right-0 p-8 opacity-[0.03] rotate-12 transition-transform group-hover:rotate-6 group-hover:scale-110">
                            <Plus className="w-32 h-32" />
                        </div>

                        <div className="relative">
                            <div className="flex items-center gap-3 mb-6">
                                <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center shadow-lg shadow-indigo-500/20">
                                    <Settings className="w-5 h-5 text-white" />
                                </div>
                                <div>
                                    <h4 className="font-bold text-white">Polymarket (CLOB)</h4>
                                    <p className="text-slate-500 text-xs mt-0.5">Central Limit Order Book API</p>
                                </div>
                            </div>

                            <form onSubmit={handleSavePolymarket} className="space-y-4">
                                <div className="space-y-1.5">
                                    <label className="text-[10px] font-bold text-slate-500 uppercase tracking-wider ml-1">API Key</label>
                                    <input
                                        type="text"
                                        required
                                        value={apiKey}
                                        onChange={(e) => setApiKey(e.target.value)}
                                        className="w-full bg-slate-950 border border-slate-800 rounded-xl px-4 py-2.5 text-sm text-white focus:outline-none focus:ring-2 focus:ring-indigo-500/50 transition-all font-mono"
                                        placeholder="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
                                    />
                                </div>
                                <div className="space-y-1.5">
                                    <label className="text-[10px] font-bold text-slate-500 uppercase tracking-wider ml-1">Secret</label>
                                    <input
                                        type="password"
                                        required
                                        value={secret}
                                        onChange={(e) => setSecret(e.target.value)}
                                        className="w-full bg-slate-950 border border-slate-800 rounded-xl px-4 py-2.5 text-sm text-white focus:outline-none focus:ring-2 focus:ring-indigo-500/50 transition-all font-mono"
                                        placeholder="••••••••••••••••••••"
                                    />
                                </div>
                                <div className="space-y-1.5">
                                    <label className="text-[10px] font-bold text-slate-500 uppercase tracking-wider ml-1">Passphrase</label>
                                    <input
                                        type="password"
                                        required
                                        value={passphrase}
                                        onChange={(e) => setPassphrase(e.target.value)}
                                        className="w-full bg-slate-950 border border-slate-800 rounded-xl px-4 py-2.5 text-sm text-white focus:outline-none focus:ring-2 focus:ring-indigo-500/50 transition-all font-mono"
                                        placeholder="••••••••"
                                    />
                                </div>
                                <div className="space-y-1.5">
                                    <label className="text-[10px] font-bold text-slate-500 uppercase tracking-wider ml-1">Proxy Address (Optional)</label>
                                    <input
                                        type="text"
                                        value={proxyAddress}
                                        onChange={(e) => setProxyAddress(e.target.value)}
                                        className="w-full bg-slate-950 border border-slate-800 rounded-xl px-4 py-2.5 text-sm text-white focus:outline-none focus:ring-2 focus:ring-indigo-500/50 transition-all font-mono"
                                        placeholder="0x..."
                                    />
                                </div>

                                {error && (
                                    <div className="flex items-center gap-2 p-3 bg-rose-500/10 border border-rose-500/20 rounded-xl text-rose-400 text-xs">
                                        <AlertCircle size={14} className="shrink-0" />
                                        <span>{error}</span>
                                    </div>
                                )}

                                <button
                                    type="submit"
                                    disabled={isLoading || !apiKey || !secret || !passphrase}
                                    className="w-full mt-2 py-3 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white font-bold rounded-xl transition-all shadow-lg shadow-indigo-500/20 flex items-center justify-center gap-2"
                                >
                                    {isLoading ? <RefreshCw className="w-5 h-5 animate-spin" /> : <Plus size={18} />}
                                    Save Integration
                                </button>
                            </form>
                        </div>
                    </div>
                </div>

                {/* Developer Settings */}
                <div className="space-y-6">
                    <h3 className="text-sm font-semibold uppercase tracking-widest text-slate-500">Developer Settings</h3>
                    <div className="bg-slate-900 border border-slate-800 p-6 rounded-3xl shadow-2xl space-y-4">
                        <div className="flex items-center justify-between">
                            <div className="flex items-center gap-3">
                                <div className="w-10 h-10 rounded-xl bg-amber-500/10 flex items-center justify-center border border-amber-500/20">
                                    <Bug className="w-5 h-5 text-amber-500" />
                                </div>
                                <div>
                                    <h4 className="font-bold text-white">Debug Mode</h4>
                                    <p className="text-slate-500 text-xs mt-0.5">Enable enhanced observability & logging</p>
                                </div>
                            </div>
                            <button
                                onClick={toggleDebugMode}
                                className={clsx(
                                    "relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none",
                                    debugMode ? "bg-indigo-600" : "bg-slate-700"
                                )}
                            >
                                <span
                                    aria-hidden="true"
                                    className={clsx(
                                        "pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out",
                                        debugMode ? "translate-x-5" : "translate-x-0"
                                    )}
                                />
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};
