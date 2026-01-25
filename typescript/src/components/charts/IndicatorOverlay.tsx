
import { useState } from "react";
import clsx from "clsx";
import { Settings, X, Plus } from "lucide-react";

export type IndicatorType = "SMA" | "EMA" | "BollingerBands" | "RSI";

export interface IndicatorConfig {
    id: string;
    type: IndicatorType;
    period: number;
    color: string;
    stdDev?: number; // For Bollinger Bands
}

interface IndicatorOverlayProps {
    activeIndicators: IndicatorConfig[];
    onAddIndicator: (config: Omit<IndicatorConfig, "id">) => void;
    onRemoveIndicator: (id: string) => void;
    onUpdateIndicator: (id: string, updates: Partial<IndicatorConfig>) => void;
}

const AVAILABLE_INDICATORS: { type: IndicatorType; label: string; defaultColor: string }[] = [
    { type: "SMA", label: "Simple Moving Average", defaultColor: "#f59e0b" },
    { type: "EMA", label: "Exponential Moving Average", defaultColor: "#3b82f6" },
    { type: "BollingerBands", label: "Bollinger Bands", defaultColor: "#8b5cf6" },
    { type: "RSI", label: "Relative Strength Index", defaultColor: "#ec4899" },
];

export function IndicatorOverlay({
    activeIndicators,
    onAddIndicator,
    onRemoveIndicator,
    onUpdateIndicator,
}: IndicatorOverlayProps) {
    const [isOpen, setIsOpen] = useState(false);

    return (
        <div className="absolute top-2 left-2 z-10 font-sans">
            <button
                onClick={() => setIsOpen(!isOpen)}
                className="flex items-center gap-1.5 px-3 py-1.5 bg-slate-800/90 hover:bg-slate-700/90 backdrop-blur border border-slate-700 rounded text-xs font-medium text-slate-300 transition-colors"
            >
                <Settings size={14} />
                Indicators
                {activeIndicators.length > 0 && (
                    <span className="bg-indigo-500 text-white text-[10px] px-1.5 rounded-full ml-1">
                        {activeIndicators.length}
                    </span>
                )}
            </button>

            {isOpen && (
                <div className="absolute top-full left-0 mt-2 w-72 bg-slate-900 border border-slate-700 rounded-lg shadow-xl overflow-hidden animate-in fade-in zoom-in-95 duration-200">
                    <div className="p-3 border-b border-slate-800 flex justify-between items-center">
                        <span className="text-xs font-bold text-slate-200 uppercase tracking-wider">
                            Active Indicators
                        </span>
                        <button onClick={() => setIsOpen(false)} className="text-slate-500 hover:text-white">
                            <X size={14} />
                        </button>
                    </div>

                    <div className="p-2 space-y-2 max-h-60 overflow-y-auto">
                        {activeIndicators.length === 0 && (
                            <div className="text-xs text-slate-500 text-center py-4">
                                No active indicators
                            </div>
                        )}

                        {activeIndicators.map((ind) => (
                            <div key={ind.id} className="bg-slate-800/50 rounded border border-slate-700/50 p-2 flex items-center justify-between group">
                                <div className="flex items-center gap-2">
                                    <div
                                        className="w-2 h-2 rounded-full"
                                        style={{ backgroundColor: ind.color }}
                                    />
                                    <div className="flex flex-col">
                                        <span className="text-xs font-medium text-slate-300">
                                            {ind.type} {ind.type === "BollingerBands" ? `(${ind.period}, ${ind.stdDev})` : `(${ind.period})`}
                                        </span>
                                    </div>
                                </div>
                                <button
                                    onClick={() => onRemoveIndicator(ind.id)}
                                    className="text-slate-500 opacity-0 group-hover:opacity-100 hover:text-rose-400 transition-all p-1"
                                >
                                    <X size={12} />
                                </button>
                            </div>
                        ))}
                    </div>

                    <div className="p-2 bg-slate-950/50 border-t border-slate-800">
                        <div className="text-[10px] text-slate-500 font-bold mb-2 px-1 uppercase">Add New</div>
                        <div className="grid grid-cols-2 gap-1">
                            {AVAILABLE_INDICATORS.map((opt) => (
                                <button
                                    key={opt.type}
                                    onClick={() => onAddIndicator({
                                        type: opt.type,
                                        period: opt.type === "RSI" ? 14 : 20,
                                        color: opt.defaultColor,
                                        stdDev: opt.type === "BollingerBands" ? 2 : undefined,
                                    })}
                                    className="text-left px-2 py-1.5 hover:bg-slate-800 rounded text-xs text-slate-400 hover:text-indigo-300 transition-colors flex items-center gap-1.5"
                                >
                                    <Plus size={10} />
                                    {opt.type}
                                </button>
                            ))}
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
