import { useMemo } from "react";
import { Activity, Play, CheckCircle, BarChart3 } from "lucide-react";

interface AlgoOrderProps {
    algo_orders: any[];
    current_step: number;
}

export function AlgoOrderWidget({ algo_orders, current_step }: AlgoOrderProps) {
    if (!algo_orders || algo_orders.length === 0) {
        return (
            <div className="flex flex-col items-center justify-center h-full text-slate-500 p-4 text-center">
                <Activity size={32} className="mb-2 opacity-20" />
                <p className="text-sm">No active algorithms running</p>
                <p className="text-xs opacity-60">Submit a TWAP, VWAP or POV order to begin</p>
            </div>
        );
    }

    return (
        <div className="flex flex-col h-full bg-slate-900 border-t border-slate-800">
            <div className="px-3 py-2 border-b border-slate-800 flex items-center justify-between">
                <span className="text-xs font-bold uppercase tracking-wider text-slate-400 flex items-center gap-2">
                    <BarChart3 size={14} className="text-indigo-400" />
                    Execution Engine
                </span>
                <span className="text-[10px] bg-indigo-500/10 text-indigo-400 px-1.5 py-0.5 rounded">
                    {algo_orders.length} Active
                </span>
            </div>
            <div className="flex-1 overflow-y-auto p-2 space-y-2">
                {algo_orders.map((algo, idx) => {
                    const type = Object.keys(algo)[0];
                    const state = algo[type];

                    const progress = state.total_quantity > 0
                        ? (state.executed_quantity / state.total_quantity) * 100
                        : 0;

                    const isFinished = state.executed_quantity >= state.total_quantity;

                    return (
                        <div key={idx} className="bg-slate-950 border border-slate-800 rounded-lg p-3 space-y-2">
                            <div className="flex items-center justify-between">
                                <span className="text-xs font-bold text-slate-200">
                                    {type} {state.side === "Bid" ? "BUY" : "SELL"}
                                </span>
                                {isFinished ? (
                                    <CheckCircle size={14} className="text-emerald-500" />
                                ) : (
                                    <Play size={12} className="text-indigo-400 animate-pulse" />
                                )}
                            </div>

                            <div className="flex justify-between text-[10px] text-slate-500">
                                <span>Progress: {progress.toFixed(1)}%</span>
                                <span>{state.executed_quantity.toFixed(1)} / {state.total_quantity.toFixed(1)}</span>
                            </div>

                            {/* Progress Bar */}
                            <div className="h-1.5 w-full bg-slate-800 rounded-full overflow-hidden">
                                <div
                                    className="h-full bg-indigo-500 transition-all duration-500"
                                    style={{ width: `${progress}%` }}
                                />
                            </div>

                            {state.end_step && (
                                <div className="text-[9px] text-slate-600 flex justify-between">
                                    <span>Steps: {current_step} / {state.end_step}</span>
                                    <span>Remaining: {Math.max(0, state.end_step - current_step)}</span>
                                </div>
                            )}
                        </div>
                    );
                })}
            </div>
        </div>
    );
}
