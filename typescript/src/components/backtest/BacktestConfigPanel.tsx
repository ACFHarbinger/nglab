
import { Calendar, DollarSign, BarChart } from "lucide-react";

export interface BacktestConfig {
    startDate: string;
    endDate: string;
    initialCapital: number;
    symbol: string;
    strategyId?: string;
    timeframe: string;
}

interface BacktestConfigProps {
    config: BacktestConfig;
    onChange: (c: BacktestConfig) => void;
    onRun: () => void;
    isRunning: boolean;
}

export function BacktestConfigPanel({ config, onChange, onRun, isRunning }: BacktestConfigProps) {
    return (
        <div className="bg-slate-900 rounded-xl border border-slate-800 p-4 space-y-4 h-fit">
            <h3 className="text-sm font-bold text-slate-400 uppercase tracking-wider mb-2">Configuration</h3>

            {/* Date Range */}
            <div className="space-y-2">
                <label className="text-xs font-semibold text-slate-500">Date Range</label>
                <div className="grid grid-cols-2 gap-2">
                    <div className="relative">
                        <input
                            type="date"
                            value={config.startDate}
                            onChange={(e) => onChange({ ...config, startDate: e.target.value })}
                            className="w-full bg-slate-800 border border-slate-700 rounded px-2 py-1.5 text-xs text-slate-200 outline-none focus:border-indigo-500"
                        />
                    </div>
                    <div className="relative">
                        <input
                            type="date"
                            value={config.endDate}
                            onChange={(e) => onChange({ ...config, endDate: e.target.value })}
                            className="w-full bg-slate-800 border border-slate-700 rounded px-2 py-1.5 text-xs text-slate-200 outline-none focus:border-indigo-500"
                        />
                    </div>
                </div>
            </div>

            {/* Capital & Symbol */}
            <div className="grid grid-cols-2 gap-2">
                <div className="space-y-1">
                    <label className="text-xs font-semibold text-slate-500 flex items-center gap-1">
                        <DollarSign size={10} /> Initial Capital
                    </label>
                    <input
                        type="number"
                        value={config.initialCapital}
                        onChange={(e) => onChange({ ...config, initialCapital: parseFloat(e.target.value) })}
                        className="w-full bg-slate-800 border border-slate-700 rounded px-2 py-1.5 text-xs text-slate-200 outline-none focus:border-indigo-500"
                    />
                </div>
                <div className="space-y-1">
                    <label className="text-xs font-semibold text-slate-500 flex items-center gap-1">
                        <BarChart size={10} /> Symbol
                    </label>
                    <select
                        value={config.symbol}
                        onChange={(e) => onChange({ ...config, symbol: e.target.value })}
                        className="w-full bg-slate-800 border border-slate-700 rounded px-2 py-1.5 text-xs text-slate-200 outline-none focus:border-indigo-500"
                    >
                        <option value="BTC-USDC">BTC-USDC</option>
                        <option value="ETH-USDC">ETH-USDC</option>
                        <option value="SOL-USDC">SOL-USDC</option>
                    </select>
                </div>
            </div>

            {/* Strategy Selector (Mock) */}
            <div className="space-y-1">
                <label className="text-xs font-semibold text-slate-500">Strategy</label>
                <select
                    className="w-full bg-slate-800 border border-slate-700 rounded px-2 py-1.5 text-xs text-slate-200 outline-none focus:border-indigo-500"
                >
                    <option>Simple Moving Average Crossover</option>
                    <option>RSI Mean Reversion</option>
                    <option>Bollinger Band Breakout</option>
                    <option value="custom">Custom Strategy (Builder)</option>
                </select>
            </div>

            <button
                onClick={onRun}
                disabled={isRunning}
                className="w-full py-2 bg-indigo-600 hover:bg-indigo-500 disabled:bg-slate-700 disabled:text-slate-500 rounded font-bold text-white text-sm transition-all shadow-lg shadow-indigo-500/20 active:scale-95 mt-4"
            >
                {isRunning ? "Running..." : "Run Backtest"}
            </button>

        </div>
    );
}
