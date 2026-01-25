import { useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { Shield, Play, Square, Settings2, BarChart4, TrendingUp, TrendingDown, Target } from "lucide-react";

interface MarketMakerProps {
    mm_active: boolean;
    mm_realized_pnl: number;
    inventory: number;
}

export function MarketMakerWidget({ mm_active, mm_realized_pnl, inventory }: MarketMakerProps) {
    const [config, setConfig] = useState({
        target_spread: 0.002,
        quote_size: 10.0,
        max_inventory: 100.0,
        skew_intensity: 0.5,
    });

    const [isConfiguring, setIsConfiguring] = useState(false);

    const handleToggleMM = async () => {
        try {
            if (mm_active) {
                await invoke("stop_market_maker");
            } else {
                await invoke("start_market_maker", { config });
            }
        } catch (e) {
            console.error("Failed to toggle Market Maker", e);
        }
    };

    return (
        <div className="flex flex-col h-full bg-slate-900 border-t border-slate-800">
            <div className="px-3 py-2 border-b border-slate-800 flex items-center justify-between bg-slate-900/50">
                <span className="text-xs font-bold uppercase tracking-wider text-slate-400 flex items-center gap-2">
                    <Target size={14} className="text-emerald-400" />
                    Market Maker
                </span>
                <div className="flex items-center gap-2">
                    <button
                        onClick={() => setIsConfiguring(!isConfiguring)}
                        className={`p-1 rounded transition-colors ${isConfiguring ? 'bg-slate-700 text-white' : 'text-slate-500 hover:text-slate-300'}`}
                    >
                        <Settings2 size={12} />
                    </button>
                    <button
                        onClick={handleToggleMM}
                        className={`flex items-center gap-1.5 px-2 py-0.5 rounded text-[10px] font-bold transition-all ${mm_active
                                ? "bg-rose-500/20 text-rose-400 border border-rose-500/30 hover:bg-rose-500/30"
                                : "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 hover:bg-emerald-500/30"
                            }`}
                    >
                        {mm_active ? <Square size={10} fill="currentColor" /> : <Play size={10} fill="currentColor" />}
                        {mm_active ? "STOP" : "START"}
                    </button>
                </div>
            </div>

            <div className="flex-1 overflow-y-auto p-3 space-y-4">
                {isConfiguring ? (
                    <div className="space-y-3 bg-slate-950/50 p-2 rounded border border-slate-800/50">
                        <div className="space-y-1">
                            <label className="text-[10px] uppercase text-slate-500 font-bold">Target Spread</label>
                            <input
                                type="number"
                                step="0.0001"
                                value={config.target_spread}
                                onChange={(e) => setConfig({ ...config, target_spread: parseFloat(e.target.value) })}
                                className="w-full bg-slate-900 border border-slate-800 rounded px-2 py-1 text-xs text-white"
                            />
                        </div>
                        <div className="space-y-1">
                            <label className="text-[10px] uppercase text-slate-500 font-bold">Quote Size</label>
                            <input
                                type="number"
                                value={config.quote_size}
                                onChange={(e) => setConfig({ ...config, quote_size: parseFloat(e.target.value) })}
                                className="w-full bg-slate-900 border border-slate-800 rounded px-2 py-1 text-xs text-white"
                            />
                        </div>
                        <div className="space-y-1">
                            <label className="text-[10px] uppercase text-slate-500 font-bold">Max Inventory</label>
                            <input
                                type="number"
                                value={config.max_inventory}
                                onChange={(e) => setConfig({ ...config, max_inventory: parseFloat(e.target.value) })}
                                className="w-full bg-slate-900 border border-slate-800 rounded px-2 py-1 text-xs text-white"
                            />
                        </div>
                    </div>
                ) : (
                    <>
                        {/* Stats Row */}
                        <div className="grid grid-cols-2 gap-2">
                            <div className="bg-slate-950 rounded p-2 border border-slate-800 flex flex-col items-center justify-center">
                                <span className="text-[9px] text-slate-500 uppercase font-bold">Realized PnL</span>
                                <span className={`text-sm font-mono font-bold ${mm_realized_pnl >= 0 ? "text-emerald-400" : "text-rose-400"}`}>
                                    ${mm_realized_pnl.toFixed(2)}
                                </span>
                            </div>
                            <div className="bg-slate-950 rounded p-2 border border-slate-800 flex flex-col items-center justify-center">
                                <span className="text-[9px] text-slate-500 uppercase font-bold">Current Pos</span>
                                <span className={`text-sm font-mono font-bold ${inventory >= 0 ? "text-indigo-400" : "text-amber-400"}`}>
                                    {inventory > 0 ? "+" : ""}{inventory.toFixed(1)}
                                </span>
                            </div>
                        </div>

                        {/* Inventory Bar */}
                        <div className="space-y-1.5 pt-1">
                            <div className="flex justify-between text-[10px] font-bold">
                                <span className="text-slate-500 uppercase">Inventory Risk</span>
                                <span className={Math.abs(inventory) > config.max_inventory * 0.8 ? "text-rose-400" : "text-slate-400"}>
                                    {Math.abs((inventory / config.max_inventory) * 100).toFixed(0)}%
                                </span>
                            </div>
                            <div className="h-1.5 w-full bg-slate-800 rounded-full overflow-hidden flex relative">
                                <div className="absolute left-1/2 top-0 bottom-0 w-px bg-slate-700 z-10" />
                                <div
                                    className={`h-full transition-all duration-500 ${inventory >= 0 ? 'bg-indigo-500 ml-auto mr-[50%]' : 'bg-amber-500 mr-auto ml-[50%]'}`}
                                    style={{
                                        width: `${Math.min(50, Math.abs(inventory / config.max_inventory) * 50)}%`,
                                        transform: inventory >= 0 ? 'scaleX(-1)' : 'none',
                                        transformOrigin: 'right'
                                    }}
                                />
                            </div>
                        </div>

                        {/* Simulation Status */}
                        {!mm_active && (
                            <div className="bg-amber-500/5 border border-amber-500/20 rounded p-2 text-[10px] text-amber-400 flex items-center gap-2">
                                <Shield size={12} className="shrink-0" />
                                <p>Automated market maker is currently inactive. Review configuration before starting.</p>
                            </div>
                        )}

                        {mm_active && (
                            <div className="bg-emerald-500/5 border border-emerald-500/20 rounded p-2 text-[10px] text-emerald-400 flex items-center gap-2 animate-pulse">
                                <BarChart4 size={12} className="shrink-0" />
                                <p>Engine active. Capturing spread with {config.skew_intensity * 100}% skew intensity.</p>
                            </div>
                        )}
                    </>
                )}
            </div>
        </div>
    );
}
