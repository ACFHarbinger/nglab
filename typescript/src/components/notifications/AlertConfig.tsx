import { useState } from "react";
import { Bell, X, Info, AlertTriangle, CheckCircle, Trash2 } from "lucide-react";
import clsx from "clsx";
import { invoke } from "@tauri-apps/api/core";

interface Alert {
    id: String;
    alert_type: string;
    target: string;
    value: number;
    active: bool;
}

interface AlertConfigProps {
    symbol: string;
    currentPrice: number;
    onClose: () => void;
    onAlertCreated: () => void;
}

export function AlertConfig({ symbol, currentPrice, onClose, onAlertCreated }: AlertConfigProps) {
    const [alertType, setAlertType] = useState("PriceAbove");
    const [targetValue, setTargetValue] = useState(currentPrice.toString());
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const handleCreateAlert = async () => {
        setLoading(true);
        setError(null);
        try {
            await invoke("add_alert", {
                alertType,
                target: symbol,
                value: parseFloat(targetValue)
            });
            onAlertCreated();
            onClose();
        } catch (e: any) {
            setError(e.toString());
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 space-y-4 shadow-2xl">
            <div className="flex items-center justify-between mb-2">
                <h3 className="text-lg font-bold text-white flex items-center gap-2">
                    <Bell className="text-indigo-400" size={20} />
                    Create Price Alert
                </h3>
                <button onClick={onClose} className="text-slate-500 hover:text-white transition-colors">
                    <X size={20} />
                </button>
            </div>

            <div className="space-y-4">
                <div>
                    <label className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-1.5 block">Alert Type</label>
                    <div className="grid grid-cols-2 gap-2">
                        <button
                            onClick={() => setAlertType("PriceAbove")}
                            className={clsx(
                                "py-2 px-3 rounded-lg text-sm font-medium border transition-all",
                                alertType === "PriceAbove"
                                    ? "bg-indigo-500/10 border-indigo-500 text-indigo-400"
                                    : "bg-slate-800/50 border-slate-700 text-slate-400 hover:border-slate-600"
                            )}
                        >
                            Price goes above
                        </button>
                        <button
                            onClick={() => setAlertType("PriceBelow")}
                            className={clsx(
                                "py-2 px-3 rounded-lg text-sm font-medium border transition-all",
                                alertType === "PriceBelow"
                                    ? "bg-indigo-500/10 border-indigo-500 text-indigo-400"
                                    : "bg-slate-800/50 border-slate-700 text-slate-400 hover:border-slate-600"
                            )}
                        >
                            Price goes below
                        </button>
                    </div>
                </div>

                <div>
                    <label className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-1.5 block">Target Price ({symbol})</label>
                    <div className="relative">
                        <span className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500">$</span>
                        <input
                            type="number"
                            value={targetValue}
                            onChange={(e) => setTargetValue(e.target.value)}
                            className="w-full bg-slate-950 border border-slate-800 rounded-lg py-2 pl-7 pr-3 text-white focus:outline-none focus:border-indigo-500 transition-all font-mono"
                        />
                    </div>
                    <p className="text-[10px] text-slate-500 mt-1.5 flex items-center gap-1">
                        <Info size={10} /> Current price: ${currentPrice.toFixed(4)}
                    </p>
                </div>

                {error && (
                    <div className="bg-rose-500/10 border border-rose-500/20 text-rose-400 p-3 rounded-lg text-xs flex items-center gap-2">
                        <AlertTriangle size={14} /> {error}
                    </div>
                )}

                <button
                    onClick={handleCreateAlert}
                    disabled={loading || !targetValue}
                    className="w-full py-3 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed text-white rounded-lg font-bold text-sm transition-all shadow-lg shadow-indigo-500/20"
                >
                    {loading ? "Creating..." : "Create Alert"}
                </button>
            </div>
        </div>
    );
}
