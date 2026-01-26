
import { useState } from "react";
import { Check, ChevronDown, Globe, Server, Shield, Wifi } from "lucide-react";
import clsx from "clsx";

export type ExchangeId = "polymarket" | "binance" | "kraken" | "deribit";

interface ExchangeOption {
    id: ExchangeId;
    name: string;
    status: "connected" | "disconnected" | "error";
    ping?: number;
}

interface ExchangeSelectorProps {
    selected: ExchangeId;
    onSelect: (id: ExchangeId) => void;
}

export function ExchangeSelector({ selected, onSelect }: ExchangeSelectorProps) {
    const [isOpen, setIsOpen] = useState(false);

    // Mock statuses
    const exchanges: ExchangeOption[] = [
        { id: "polymarket", name: "Polymarket", status: "connected", ping: 45 },
        { id: "binance", name: "Binance", status: "disconnected" },
        { id: "kraken", name: "Kraken", status: "disconnected" },
        { id: "deribit", name: "Deribit", status: "error" },
    ];

    const current = exchanges.find(e => e.id === selected) || exchanges[0];

    return (
        <div className="relative">
            <button
                onClick={() => setIsOpen(!isOpen)}
                className="flex items-center gap-2 bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded-lg px-3 py-2 text-sm font-medium text-slate-200 transition-colors w-48 justify-between"
            >
                <div className="flex items-center gap-2">
                    <Globe size={14} className="text-slate-400" />
                    <span>{current.name}</span>
                </div>
                <div className="flex items-center gap-2">
                    <StatusDot status={current.status} />
                    <ChevronDown size={14} className="text-slate-500" />
                </div>
            </button>

            {isOpen && (
                <div className="absolute top-full left-0 mt-2 w-64 bg-slate-900 border border-slate-700 rounded-xl shadow-xl shadow-black/50 z-50 overflow-hidden">
                    <div className="p-2 border-b border-slate-800 bg-slate-800/20">
                        <span className="text-xs font-bold text-slate-500 uppercase px-2">Data Source</span>
                    </div>
                    <div className="p-1 space-y-1">
                        {exchanges.map(ex => (
                            <button
                                key={ex.id}
                                onClick={() => {
                                    onSelect(ex.id);
                                    setIsOpen(false);
                                }}
                                className={clsx(
                                    "w-full flex items-center justify-between p-2 rounded-lg text-sm transition-colors",
                                    selected === ex.id
                                        ? "bg-indigo-500/10 text-indigo-300"
                                        : "text-slate-300 hover:bg-slate-800"
                                )}
                            >
                                <div className="flex items-center gap-3">
                                    {selected === ex.id && <Check size={14} className="text-indigo-400" />}
                                    <span className={clsx(selected !== ex.id && "ml-6")}>{ex.name}</span>
                                </div>
                                <div className="flex items-center gap-2">
                                    {ex.ping && <span className="text-xs text-slate-500 font-mono">{ex.ping}ms</span>}
                                    <StatusDot status={ex.status} />
                                </div>
                            </button>
                        ))}
                    </div>

                    <div className="p-2 border-t border-slate-800 mt-1">
                        <button className="w-full py-1.5 flex items-center justify-center gap-2 text-xs font-medium text-slate-400 hover:text-slate-200 bg-slate-800/50 hover:bg-slate-800 rounded transition-colors">
                            <Server size={12} /> Manage Connections
                        </button>
                    </div>
                </div>
            )}
        </div>
    );
}

function StatusDot({ status }: { status: string }) {
    if (status === "connected") return <div className="w-2 h-2 rounded-full bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.4)]" />;
    if (status === "error") return <div className="w-2 h-2 rounded-full bg-rose-500" />;
    return <div className="w-2 h-2 rounded-full bg-slate-600" />;
}
