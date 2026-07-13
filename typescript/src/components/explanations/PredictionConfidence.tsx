/**
 * @module components/explanations/PredictionConfidence
 * @description Displays prediction confidence intervals and out-of-distribution indicators.
 */

import { AlertTriangle, CheckCircle, Activity, TrendingUp, TrendingDown, Minus } from "lucide-react";
import clsx from "clsx";

interface PredictionResult {
  prediction: number;
  confidence: number;        // 0-1 confidence score
  confidenceInterval: {
    lower: number;
    upper: number;
  };
  isOOD: boolean;            // out-of-distribution flag
  oodScore?: number;         // 0-1 scale
  direction: "up" | "down" | "neutral";
  timestamp: number;
}

interface PredictionConfidenceProps {
  result: PredictionResult | null;
  isLoading?: boolean;
}

// Mock prediction for demo
export const MOCK_PREDICTION: PredictionResult = {
  prediction: 0.032,
  confidence: 0.78,
  confidenceInterval: {
    lower: 0.015,
    upper: 0.048,
  },
  isOOD: false,
  oodScore: 0.12,
  direction: "up",
  timestamp: Date.now(),
};

function formatPercent(value: number): string {
  return `${(value * 100).toFixed(2)}%`;
}

function formatConfidenceBar(value: number): string {
  return `${(value * 100).toFixed(0)}%`;
}

export function PredictionConfidence({ result, isLoading }: PredictionConfidenceProps) {
  if (isLoading) {
    return (
      <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-6 h-48 flex items-center justify-center">
        <div className="animate-pulse flex items-center gap-2 text-slate-500">
          <Activity size={20} />
          Generating prediction...
        </div>
      </div>
    );
  }

  if (!result) return null;

  const confidenceLevel = 
    result.confidence >= 0.8 ? "high" :
    result.confidence >= 0.5 ? "medium" : "low";

  return (
    <div className="space-y-4">
      {/* Main Prediction Card */}
      <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-6">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-3">
            {result.direction === "up" && <TrendingUp className="w-6 h-6 text-emerald-400" />}
            {result.direction === "down" && <TrendingDown className="w-6 h-6 text-red-400" />}
            {result.direction === "neutral" && <Minus className="w-6 h-6 text-slate-400" />}
            <h4 className="font-semibold text-slate-200">Prediction</h4>
          </div>
          
          {/* OOD Indicator */}
          <div className={clsx(
            "flex items-center gap-2 px-3 py-1 rounded-full text-xs font-medium",
            result.isOOD 
              ? "bg-amber-500/20 text-amber-400 border border-amber-500/30" 
              : "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30"
          )}>
            {result.isOOD ? (
              <>
                <AlertTriangle size={12} />
                Out-of-Distribution
              </>
            ) : (
              <>
                <CheckCircle size={12} />
                In-Distribution
              </>
            )}
          </div>
        </div>

        {/* Prediction Value */}
        <div className="flex items-baseline gap-4 mb-6">
          <span className={clsx(
            "text-4xl font-bold font-mono",
            result.direction === "up" ? "text-emerald-400" :
            result.direction === "down" ? "text-red-400" : "text-slate-300"
          )}>
            {result.direction === "up" ? "+" : ""}{formatPercent(result.prediction)}
          </span>
          <span className="text-slate-500 text-sm">
            Expected return
          </span>
        </div>

        {/* Confidence Interval Visualization */}
        <div className="mb-6">
          <div className="flex justify-between text-xs text-slate-500 mb-2">
            <span>Confidence Interval (95%)</span>
            <span>{formatPercent(result.confidenceInterval.lower)} — {formatPercent(result.confidenceInterval.upper)}</span>
          </div>
          <div className="relative h-8 bg-slate-800 rounded-lg overflow-hidden">
            {/* Confidence interval bar */}
            <div 
              className="absolute h-full bg-indigo-500/30 border-l-2 border-r-2 border-indigo-500"
              style={{
                left: `${((result.confidenceInterval.lower + 0.1) / 0.2) * 100}%`,
                right: `${100 - ((result.confidenceInterval.upper + 0.1) / 0.2) * 100}%`,
              }}
            />
            {/* Prediction point */}
            <div 
              className="absolute top-0 bottom-0 w-1 bg-white rounded"
              style={{
                left: `${((result.prediction + 0.1) / 0.2) * 100}%`,
              }}
            />
            {/* Center line */}
            <div className="absolute left-1/2 top-0 bottom-0 w-px bg-slate-600" />
          </div>
          <div className="flex justify-between text-xs text-slate-600 mt-1">
            <span>-10%</span>
            <span>0%</span>
            <span>+10%</span>
          </div>
        </div>

        {/* Confidence Score */}
        <div>
          <div className="flex justify-between text-sm mb-2">
            <span className="text-slate-400">Model Confidence</span>
            <span className={clsx(
              "font-medium",
              confidenceLevel === "high" ? "text-emerald-400" :
              confidenceLevel === "medium" ? "text-amber-400" : "text-red-400"
            )}>
              {formatConfidenceBar(result.confidence)}
            </span>
          </div>
          <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
            <div 
              className={clsx(
                "h-full rounded-full transition-all",
                confidenceLevel === "high" ? "bg-emerald-500" :
                confidenceLevel === "medium" ? "bg-amber-500" : "bg-red-500"
              )}
              style={{ width: `${result.confidence * 100}%` }}
            />
          </div>
        </div>
      </div>

      {/* OOD Details (if applicable) */}
      {result.oodScore !== undefined && (
        <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <AlertTriangle size={16} className="text-amber-400" />
              <span className="text-sm text-slate-400">Distribution Distance</span>
            </div>
            <span className="text-sm font-mono text-slate-300">{(result.oodScore * 100).toFixed(1)}%</span>
          </div>
          <div className="mt-2 h-1.5 bg-slate-800 rounded-full overflow-hidden">
            <div 
              className={clsx(
                "h-full rounded-full",
                result.oodScore > 0.5 ? "bg-red-500" : 
                result.oodScore > 0.2 ? "bg-amber-500" : "bg-emerald-500"
              )}
              style={{ width: `${result.oodScore * 100}%` }}
            />
          </div>
          <p className="text-xs text-slate-500 mt-2">
            {result.oodScore < 0.2 
              ? "Input is well within training distribution" 
              : result.oodScore < 0.5 
              ? "Input is near the edge of training distribution"
              : "Input may be out of distribution - treat prediction with caution"}
          </p>
        </div>
      )}
    </div>
  );
}
