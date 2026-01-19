import React, { useState, useEffect } from "react";
import { invoke } from "@tauri-apps/api/core";
import {
    Lock,
    Unlock,
    Plus,
    Trash2,
    Eye,
    EyeOff,
    Key,
    ShieldCheck,
    Search,
    RefreshCw,
    AlertCircle
} from "lucide-react";
import clsx from "clsx";

interface VaultSummary {
    id: number;
    label: string;
    created_at: string;
}


const VaultTab: React.FC = () => {
    const [isUnlocked, setIsUnlocked] = useState(false);
    const [masterPassword, setMasterPassword] = useState("");
    const [summaries, setSummaries] = useState<VaultSummary[]>([]);
    const [searchQuery, setSearchQuery] = useState("");
    const [newLabel, setNewLabel] = useState("");
    const [newValue, setNewValue] = useState("");
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [decryptedValues, setDecryptedValues] = useState<Record<number, string>>({});
    const [showValues, setShowValues] = useState<Record<number, boolean>>({});

    useEffect(() => {
        checkUnlockStatus();
    }, []);

    const checkUnlockStatus = async () => {
        try {
            const status = await invoke<boolean>("is_vault_unlocked");
            setIsUnlocked(status);
            if (status) {
                fetchSummaries();
            }
        } catch (err) {
            console.error("Failed to check unlock status", err);
        }
    };

    const fetchSummaries = async () => {
        try {
            const response: any = await invoke("list_vault_secrets");
            if (response.success) {
                setSummaries(response.data || []);
            }
        } catch (err) {
            setError("Failed to fetch entries");
        }
    };

    const handleUnlock = async (e: React.FormEvent) => {
        e.preventDefault();
        setIsLoading(true);
        setError(null);
        try {
            const response: any = await invoke("unlock_vault", { password: masterPassword });
            if (response.success) {
                setIsUnlocked(true);
                setMasterPassword("");
                fetchSummaries();
            } else {
                setError(response.message);
            }
        } catch (err) {
            setError("Failed to unlock vault");
        } finally {
            setIsLoading(false);
        }
    };

    const handleLock = async () => {
        try {
            await invoke("lock_vault");
            setIsUnlocked(false);
            setSummaries([]);
            setDecryptedValues({});
            setShowValues({});
        } catch (err) {
            console.error("Failed to lock vault", err);
        }
    };

    const handleAddSecret = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!newLabel || !newValue) return;
        setIsLoading(true);
        try {
            const response: any = await invoke("add_vault_secret", { label: newLabel, value: newValue });
            if (response.success) {
                setNewLabel("");
                setNewValue("");
                fetchSummaries();
            } else {
                setError(response.message);
            }
        } catch (err) {
            setError("Failed to add secret");
        } finally {
            setIsLoading(false);
        }
    };

    const toggleViewSecret = async (id: number) => {
        if (showValues[id]) {
            setShowValues(prev => ({ ...prev, [id]: false }));
            return;
        }

        if (decryptedValues[id]) {
            setShowValues(prev => ({ ...prev, [id]: true }));
            return;
        }

        try {
            const response: any = await invoke("get_vault_secret", { id });
            if (response.success && response.data) {
                setDecryptedValues(prev => ({ ...prev, [id]: response.data.value }));
                setShowValues(prev => ({ ...prev, [id]: true }));
            } else {
                setError(response.message || "Decryption failed. Vault might be locked.");
            }
        } catch (err) {
            setError("Failed to retrieve secret");
        }
    };

    const handleDeleteSecret = async (id: number) => {
        if (!confirm("Are you sure you want to delete this secret?")) return;
        try {
            const response: any = await invoke("delete_vault_secret", { id });
            if (response.success) {
                setSummaries(prev => prev.filter(e => e.id !== id));
                const newDecrypted = { ...decryptedValues };
                delete newDecrypted[id];
                setDecryptedValues(newDecrypted);
            }
        } catch (err) {
            setError("Failed to delete secret");
        }
    };

    const filteredSummaries = summaries.filter(e =>
        e.label.toLowerCase().includes(searchQuery.toLowerCase())
    );

    if (!isUnlocked) {
        return (
            <div className="flex-1 flex flex-col items-center justify-center p-6 bg-slate-900/50">
                <div className="w-full max-w-md p-8 bg-slate-900 rounded-2xl border border-slate-800 shadow-2xl">
                    <div className="flex flex-col items-center mb-8">
                        <div className="w-16 h-16 rounded-2xl bg-indigo-500/10 flex items-center justify-center mb-4">
                            <Lock className="w-8 h-8 text-indigo-400" />
                        </div>
                        <h2 className="text-2xl font-bold text-white">Unlock Vault</h2>
                        <p className="text-slate-400 text-center mt-2">
                            Enter your master password to access your encrypted secrets.
                        </p>
                    </div>

                    <form onSubmit={handleUnlock} className="space-y-4">
                        <div>
                            <label className="block text-sm font-medium text-slate-300 mb-1.5">
                                Master Password
                            </label>
                            <div className="relative">
                                <input
                                    type="password"
                                    value={masterPassword}
                                    onChange={(e) => setMasterPassword(e.target.value)}
                                    className="w-full bg-slate-950 border border-slate-800 rounded-xl px-4 py-3 text-white placeholder-slate-600 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 transition-all"
                                    placeholder="••••••••"
                                    autoFocus
                                />
                                <Key className="absolute right-4 top-3.5 w-5 h-5 text-slate-600" />
                            </div>
                        </div>

                        {error && (
                            <div className="flex items-center gap-2 text-red-400 text-sm bg-red-400/10 p-3 rounded-lg border border-red-400/20">
                                <AlertCircle className="w-4 h-4 shrink-0" />
                                <span>{error}</span>
                            </div>
                        )}

                        <button
                            type="submit"
                            disabled={isLoading || !masterPassword}
                            className="w-full bg-indigo-500 hover:bg-indigo-600 disabled:opacity-50 text-white font-semibold py-3 rounded-xl transition-all shadow-lg shadow-indigo-500/20 flex items-center justify-center gap-2"
                        >
                            {isLoading ? <RefreshCw className="w-5 h-5 animate-spin" /> : <Unlock className="w-5 h-5" />}
                            {isLoading ? "Unlocking..." : "Unlock Vault"}
                        </button>
                    </form>

                    <p className="text-xs text-slate-500 text-center mt-6">
                        <ShieldCheck className="w-3 h-3 inline mr-1 mb-0.5" />
                        Secrets are encrypted on-disk using AES-256 (SQLCipher).
                    </p>
                </div>
            </div>
        );
    }

    return (
        <div className="flex-1 flex flex-col p-6 gap-6 overflow-hidden">
            {/* Header Controls */}
            <div className="flex items-center justify-between">
                <div className="flex items-center gap-4">
                    <div className="relative w-80">
                        <input
                            type="text"
                            value={searchQuery}
                            onChange={(e) => setSearchQuery(e.target.value)}
                            placeholder="Search secrets..."
                            className="w-full bg-slate-900 border border-slate-800 rounded-xl px-10 py-2.5 text-sm text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 transition-all"
                        />
                        <Search className="absolute left-3.5 top-2.5 w-4.5 h-4.5 text-slate-500" />
                    </div>
                    <button
                        onClick={fetchSummaries}
                        className="p-2.5 rounded-xl border border-slate-800 bg-slate-900 text-slate-400 hover:text-white hover:bg-slate-800 transition-all"
                    >
                        <RefreshCw className="w-4.5 h-4.5" />
                    </button>
                </div>

                <button
                    onClick={handleLock}
                    className="flex items-center gap-2 px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 text-sm font-medium rounded-xl transition-all border border-slate-700"
                >
                    <Lock size={16} /> Lock Vault
                </button>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 flex-1 overflow-hidden">
                {/* Left: Secrets List */}
                <div className="lg:col-span-2 flex flex-col gap-4 overflow-y-auto pr-2 custom-scrollbar">
                    {filteredSummaries.length === 0 ? (
                        <div className="flex flex-col items-center justify-center py-20 bg-slate-900/30 rounded-2xl border border-dashed border-slate-800">
                            <Key className="w-12 h-12 text-slate-700 mb-4" />
                            <p className="text-slate-500">No secrets found in vault.</p>
                        </div>
                    ) : (
                        filteredSummaries.map((summary) => (
                            <div
                                key={summary.id}
                                className="p-4 bg-slate-900 rounded-xl border border-slate-800 hover:border-slate-700 transition-all flex items-center justify-between group"
                            >
                                <div className="flex-1 min-w-0">
                                    <div className="flex items-center gap-2 mb-1">
                                        <h3 className="font-semibold text-white truncate">{summary.label}</h3>
                                        <span className="text-[10px] text-slate-600 bg-slate-950 px-1.5 py-0.5 rounded border border-slate-800 uppercase tracking-wider">
                                            ID: {summary.id}
                                        </span>
                                    </div>
                                    <div className="flex items-center gap-2">
                                        <div className="bg-slate-950 px-3 py-1.5 rounded-lg border border-slate-800 flex-1 flex items-center justify-between">
                                            <span className={clsx(
                                                "text-sm font-mono truncate text-indigo-400",
                                                !showValues[summary.id] && "opacity-30 blur-[2px]"
                                            )}>
                                                {showValues[summary.id] ? decryptedValues[summary.id] : "••••••••••••••••"}
                                            </span>
                                            <button
                                                onClick={() => toggleViewSecret(summary.id)}
                                                className="text-slate-500 hover:text-white transition-colors ml-2"
                                            >
                                                {showValues[summary.id] ? <EyeOff size={14} /> : <Eye size={14} />}
                                            </button>
                                        </div>
                                    </div>
                                </div>

                                <div className="flex items-center gap-2 ml-4 opacity-0 group-hover:opacity-100 transition-opacity">
                                    <button
                                        onClick={() => handleDeleteSecret(summary.id)}
                                        className="p-2 text-slate-500 hover:text-red-400 hover:bg-red-400/10 rounded-lg transition-all"
                                    >
                                        <Trash2 size={18} />
                                    </button>
                                </div>
                            </div>
                        ))
                    )}
                </div>

                {/* Right: Add New */}
                <div className="flex flex-col gap-4">
                    <div className="p-6 bg-slate-900 rounded-2xl border border-slate-800 shadow-xl">
                        <div className="flex items-center gap-2 mb-6">
                            <div className="w-8 h-8 rounded-lg bg-indigo-500/10 flex items-center justify-center">
                                <Plus className="w-4 h-4 text-indigo-400" />
                            </div>
                            <h2 className="text-lg font-bold text-white">Add New Secret</h2>
                        </div>

                        <form onSubmit={handleAddSecret} className="space-y-4">
                            <div>
                                <label className="block text-xs font-medium text-slate-400 mb-1.5 uppercase tracking-wider">
                                    Label / Description
                                </label>
                                <input
                                    type="text"
                                    value={newLabel}
                                    onChange={(e) => setNewLabel(e.target.value)}
                                    placeholder="e.g. Binance API Key"
                                    className="w-full bg-slate-950 border border-slate-800 rounded-xl px-4 py-2.5 text-sm text-white placeholder-slate-700 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 transition-all"
                                />
                            </div>

                            <div>
                                <label className="block text-xs font-medium text-slate-400 mb-1.5 uppercase tracking-wider">
                                    Secret Value
                                </label>
                                <textarea
                                    value={newValue}
                                    onChange={(e) => setNewValue(e.target.value)}
                                    placeholder="Paste your secret here..."
                                    rows={4}
                                    className="w-full bg-slate-950 border border-slate-800 rounded-xl px-4 py-2.5 text-sm text-white placeholder-slate-700 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 transition-all resize-none font-mono"
                                />
                            </div>

                            <button
                                type="submit"
                                disabled={isLoading || !newLabel || !newValue}
                                className="w-full bg-slate-800 hover:bg-indigo-600 disabled:opacity-50 text-white font-semibold py-3 rounded-xl transition-all border border-slate-700 flex items-center justify-center gap-2 group"
                            >
                                <Plus className="w-4 h-4 group-hover:scale-110 transition-transform" />
                                Add to Vault
                            </button>
                        </form>
                    </div>

                    <div className="p-4 bg-indigo-500/5 rounded-2xl border border-indigo-500/10">
                        <div className="flex gap-3">
                            <ShieldCheck className="w-5 h-5 text-indigo-400 shrink-0 mt-0.5" />
                            <div>
                                <h4 className="text-sm font-semibold text-indigo-300">Data Persistence</h4>
                                <p className="text-xs text-indigo-300/60 mt-1 leading-relaxed">
                                    Your vault is stored in the local AppData directory. The entire database file is encrypted with SQLCipher, providing military-grade protection for your secrets.
                                </p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default VaultTab;
