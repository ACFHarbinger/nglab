import React from 'react';
import { BarChart2, ShieldAlert, TrendingDown } from 'lucide-react';

// Mock data structure matching backend MonteCarloResult
interface SimulationResult {
    meanPrice: number;
    stdDev: number;
    var95: number;
    cvar95: number;
    paths: number[][];
}

interface ScenarioDashboardProps {
    result?: SimulationResult;
    isRunning?: boolean;
}

export const ScenarioDashboard: React.FC<ScenarioDashboardProps> = ({ result, isRunning }) => {
    if (!result && !isRunning) {
        return (
            <div className="flex flex-col h-full items-center justify-center text-slate-500 bg-slate-900 rounded-lg p-8 border border-slate-800 border-dashed">
                <BarChart2 size={48} className="mb-4 text-slate-700" />
                <p>No simulation results available.</p>
                <p className="text-sm">Run a scenario to see risk metrics.</p>
            </div>
        );
    }

    if (isRunning) {
        return (
            <div className="flex flex-col h-full items-center justify-center text-blue-400 bg-slate-900 rounded-lg p-8">
                <div className="w-12 h-12 border-4 border-blue-600 border-t-transparent rounded-full animate-spin mb-4"></div>
                <p>Running Monte Carlo Simulation...</p>
            </div>
        );
    }

    if (!result) return null;

    return (
        <div className="flex flex-col gap-4 h-full bg-slate-900 p-4 rounded-lg">
            <h2 className="text-xl font-bold text-white mb-2">Risk Analysis Results</h2>

            {/* Key Metrics Cards */}
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                <div className="bg-slate-800 p-4 rounded border border-slate-700">
                    <div className="text-slate-400 text-xs uppercase font-bold mb-1">Expected Price</div>
                    <div className="text-2xl font-mono text-white">${result.meanPrice.toFixed(2)}</div>
                    <div className="text-xs text-slate-500">Mean of 1000 paths</div>
                </div>
                <div className="bg-slate-800 p-4 rounded border border-slate-700">
                    <div className="text-slate-400 text-xs uppercase font-bold mb-1">Volatility (Std)</div>
                    <div className="text-2xl font-mono text-yellow-400">{result.stdDev.toFixed(2)}</div>
                </div>
                <div className="bg-red-900/20 p-4 rounded border border-red-900/50">
                    <div className="text-red-300 text-xs uppercase font-bold mb-1 flex items-center gap-1">
                        <ShieldAlert size={12} /> VaR (95%)
                    </div>
                    <div className="text-2xl font-mono text-red-400">-${result.var95.toFixed(2)}</div>
                    <div className="text-xs text-red-300/60">Value at Risk</div>
                </div>
                <div className="bg-red-900/20 p-4 rounded border border-red-900/50">
                    <div className="text-red-300 text-xs uppercase font-bold mb-1 flex items-center gap-1">
                        <TrendingDown size={12} /> CVaR (95%)
                    </div>
                    <div className="text-2xl font-mono text-red-500">-${result.cvar95.toFixed(2)}</div>
                    <div className="text-xs text-red-300/60">Expected Shortfall</div>
                </div>
            </div>

            {/* Distribution Chart Placeholder */}
            {/* In a real implementation, use Lightweight Charts or Recharts here */}
            <div className="flex-1 bg-slate-800 rounded border border-slate-700 p-4 flex flex-col items-center justify-center text-slate-500">
                <p>[Price Distribution Histogram Placeholder]</p>
                <div className="w-full h-32 flex items-end justify-center gap-1 mt-4 opacity-50">
                    {/* Fake histogram bars */}
                    {[10, 20, 45, 80, 120, 150, 110, 70, 40, 20, 10].map((h, i) => (
                        <div key={i} style={{ height: `${h / 2}px` }} className="w-8 bg-blue-500 rounded-t"></div>
                    ))}
                </div>
            </div>
        </div>
    );
};
