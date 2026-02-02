
import { Activity, Zap, TrendingUp, BarChart2, Layers, ArrowUpDown } from "lucide-react";
import clsx from "clsx";

/**
 * @module components/features/FeatureCatalog
 * @description Displays available features with their definitions, statistics, and formulas.
 */

interface FeatureDefinition {
  id: string;
  name: string;
  category: "momentum" | "volatility" | "trend" | "microstructure" | "volume";
  description: string;
  formula?: string;
  parameters: string[];
  // Feature statistics (mock data for now)
  stats?: {
    mean: number;
    std: number;
    min: number;
    max: number;
    correlation: number; // with target
  };
}

const AVAILABLE_FEATURES: FeatureDefinition[] = [
  {
    id: "log_ret",
    name: "Log Returns",
    category: "momentum",
    description: "Logarithmic returns of the close price.",
    formula: "ln(Close_t / Close_{t-1})",
    parameters: [],
    stats: { mean: 0.0002, std: 0.012, min: -0.15, max: 0.12, correlation: 0.65 },
  },
  {
    id: "sma_diff",
    name: "SMA Divergence",
    category: "trend",
    description: "Difference between short-term and long-term moving averages.",
    formula: "SMA(10) - SMA(30)",
    parameters: ["short_window=10", "long_window=30"],
    stats: { mean: -2.5, std: 15.8, min: -85, max: 72, correlation: 0.42 },
  },
  {
    id: "rsi",
    name: "Relative Strength Index",
    category: "momentum",
    description: "Momentum oscillator measuring speed and change of price movements.",
    formula: "100 - (100 / (1 + RS))",
    parameters: ["window=14"],
    stats: { mean: 52.3, std: 18.2, min: 5, max: 98, correlation: 0.38 },
  },
  {
    id: "volatility",
    name: "Rolling Volatility",
    category: "volatility",
    description: "Standard deviation of returns over a usage window.",
    formula: "std(returns, window=20)",
    parameters: ["window=20"],
    stats: { mean: 0.018, std: 0.008, min: 0.003, max: 0.085, correlation: -0.22 },
  },
  {
    id: "imbalance",
    name: "Order Book Imbalance",
    category: "microstructure",
    description: "Normalized difference between bid and ask volumes at best levels.",
    formula: "(BidVol - AskVol) / (BidVol + AskVol)",
    parameters: ["depth=1"],
    stats: { mean: 0.02, std: 0.35, min: -1.0, max: 1.0, correlation: 0.78 },
  },
  {
    id: "spread",
    name: "Bid-Ask Spread",
    category: "microstructure",
    description: "Difference between best ask and best bid prices.",
    formula: "Ask_0 - Bid_0",
    parameters: [],
    stats: { mean: 0.01, std: 0.005, min: 0.001, max: 0.08, correlation: -0.15 },
  },
  {
    id: "vwap",
    name: "VWAP",
    category: "volume",
    description: "Volume Weighted Average Price.",
    formula: "cumsum(Price * Volume) / cumsum(Volume)",
    parameters: [],
    stats: { mean: 45250.5, std: 1250.3, min: 42000, max: 48500, correlation: 0.55 },
  }
];

function formatStat(value: number): string {
  if (Math.abs(value) >= 1000) return value.toFixed(0);
  if (Math.abs(value) >= 1) return value.toFixed(2);
  if (Math.abs(value) >= 0.01) return value.toFixed(3);
  return value.toExponential(2);
}

export function FeatureCatalog() {
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {AVAILABLE_FEATURES.map((feature) => (
          <div
            key={feature.id}
            className="bg-slate-900/50 border border-slate-800 rounded-xl p-4 hover:border-indigo-500/50 transition-colors group"
          >
            <div className="flex items-start justify-between mb-3">
              <div className="flex items-center gap-3">
                <div className={clsx("p-2 rounded-lg bg-opacity-20", {
                    "bg-blue-500 text-blue-400": feature.category === "momentum",
                    "bg-purple-500 text-purple-400": feature.category === "trend",
                    "bg-orange-500 text-orange-400": feature.category === "volatility",
                    "bg-emerald-500 text-emerald-400": feature.category === "microstructure",
                    "bg-cyan-500 text-cyan-400": feature.category === "volume",
                })}>
                  {feature.category === "momentum" && <Activity className="w-5 h-5" />}
                  {feature.category === "trend" && <TrendingUp className="w-5 h-5" />}
                  {feature.category === "volatility" && <Zap className="w-5 h-5" />}
                  {feature.category === "microstructure" && <Layers className="w-5 h-5" />}
                  {feature.category === "volume" && <BarChart2 className="w-5 h-5" />}
                </div>
                <div>
                  <h3 className="font-semibold text-slate-200 group-hover:text-indigo-400 transition-colors">
                    {feature.name}
                  </h3>
                  <span className="text-xs font-mono text-slate-500 uppercase tracking-wider">
                    {feature.category}
                  </span>
                </div>
              </div>
              {/* Correlation indicator */}
              {feature.stats && (
                <div className={clsx(
                  "flex items-center gap-1 px-2 py-1 rounded text-xs font-medium",
                  feature.stats.correlation >= 0.5 ? "bg-emerald-500/20 text-emerald-400" :
                  feature.stats.correlation >= 0.3 ? "bg-amber-500/20 text-amber-400" :
                  feature.stats.correlation >= 0 ? "bg-slate-700 text-slate-400" :
                  "bg-red-500/20 text-red-400"
                )}>
                  <ArrowUpDown size={10} />
                  {(feature.stats.correlation * 100).toFixed(0)}%
                </div>
              )}
            </div>

            <p className="text-sm text-slate-400 mb-3 line-clamp-2">
              {feature.description}
            </p>

            {feature.formula && (
               <div className="bg-slate-950 rounded px-3 py-2 mb-3 font-mono text-xs text-slate-500 border border-slate-800/50">
                 {feature.formula}
               </div>
            )}

            {/* Statistics grid */}
            {feature.stats && (
              <div className="grid grid-cols-4 gap-1 mb-3">
                <div className="bg-slate-800/50 p-2 rounded text-center">
                  <div className="text-[10px] text-slate-500 uppercase">μ</div>
                  <div className="text-xs font-mono text-slate-300">{formatStat(feature.stats.mean)}</div>
                </div>
                <div className="bg-slate-800/50 p-2 rounded text-center">
                  <div className="text-[10px] text-slate-500 uppercase">σ</div>
                  <div className="text-xs font-mono text-slate-300">{formatStat(feature.stats.std)}</div>
                </div>
                <div className="bg-slate-800/50 p-2 rounded text-center">
                  <div className="text-[10px] text-slate-500 uppercase">min</div>
                  <div className="text-xs font-mono text-slate-300">{formatStat(feature.stats.min)}</div>
                </div>
                <div className="bg-slate-800/50 p-2 rounded text-center">
                  <div className="text-[10px] text-slate-500 uppercase">max</div>
                  <div className="text-xs font-mono text-slate-300">{formatStat(feature.stats.max)}</div>
                </div>
              </div>
            )}

            <div className="flex flex-wrap gap-2 mt-auto">
              {feature.parameters.map((param) => (
                <span key={param} className="px-2 py-1 rounded bg-slate-800 text-xs text-slate-400">
                  {param}
                </span>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
