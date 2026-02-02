/**
 * @module components/features/FeatureValidation
 * @description Feature validation tools including distribution histogram and stationarity tests.
 */

import { CheckCircle, AlertTriangle, XCircle, TrendingDown } from "lucide-react";
import clsx from "clsx";

interface FeatureValidationResult {
  featureName: string;
  distribution: {
    bins: { x: number; count: number }[];
    skewness: number;
    kurtosis: number;
  };
  stationarity: {
    adfStatistic: number;
    pValue: number;
    isStationary: boolean;
  };
  outliers: {
    count: number;
    percentage: number;
    threshold: number;
  };
  missingness: {
    count: number;
    percentage: number;
  };
}

interface FeatureValidationProps {
  result: FeatureValidationResult | null;
  isLoading?: boolean;
}

// Mock result for demo
export const MOCK_VALIDATION_RESULT: FeatureValidationResult = {
  featureName: "imbalance",
  distribution: {
    bins: [
      { x: -1.0, count: 5 },
      { x: -0.75, count: 12 },
      { x: -0.5, count: 28 },
      { x: -0.25, count: 85 },
      { x: 0, count: 142 },
      { x: 0.25, count: 78 },
      { x: 0.5, count: 32 },
      { x: 0.75, count: 15 },
      { x: 1.0, count: 3 },
    ],
    skewness: 0.12,
    kurtosis: 2.8,
  },
  stationarity: {
    adfStatistic: -4.52,
    pValue: 0.001,
    isStationary: true,
  },
  outliers: {
    count: 23,
    percentage: 0.58,
    threshold: 3.0,
  },
  missingness: {
    count: 0,
    percentage: 0,
  },
};

export function FeatureValidation({ result, isLoading }: FeatureValidationProps) {
  if (isLoading) {
    return (
      <div className="h-64 flex items-center justify-center text-slate-500">
        <div className="animate-pulse">Validating features...</div>
      </div>
    );
  }

  if (!result) return null;

  const maxCount = Math.max(...result.distribution.bins.map(b => b.count));

  return (
    <div className="space-y-6">
      {/* Distribution Histogram */}
      <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-5">
        <div className="flex items-center justify-between mb-4">
          <h4 className="font-semibold text-slate-200">Distribution</h4>
          <div className="flex gap-4 text-xs">
            <span className="text-slate-500">
              Skew: <span className="text-slate-300 font-mono">{result.distribution.skewness.toFixed(2)}</span>
            </span>
            <span className="text-slate-500">
              Kurt: <span className="text-slate-300 font-mono">{result.distribution.kurtosis.toFixed(2)}</span>
            </span>
          </div>
        </div>

        {/* Simple bar chart */}
        <div className="flex items-end gap-1 h-32">
          {result.distribution.bins.map((bin, i) => (
            <div key={i} className="flex-1 flex flex-col items-center gap-1">
              <div
                className="w-full bg-indigo-500 rounded-t transition-all hover:bg-indigo-400"
                style={{ height: `${(bin.count / maxCount) * 100}%` }}
                title={`${bin.x}: ${bin.count}`}
              />
              <span className="text-[10px] text-slate-500 font-mono">{bin.x}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Validation Checks Grid */}
      <div className="grid grid-cols-2 gap-4">
        {/* Stationarity */}
        <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-4">
          <div className="flex items-center gap-3 mb-3">
            {result.stationarity.isStationary ? (
              <CheckCircle className="w-5 h-5 text-emerald-400" />
            ) : (
              <XCircle className="w-5 h-5 text-red-400" />
            )}
            <h4 className="font-semibold text-slate-200">Stationarity (ADF)</h4>
          </div>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-slate-500">ADF Statistic:</span>
              <span className="text-slate-300 font-mono">{result.stationarity.adfStatistic.toFixed(2)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">p-value:</span>
              <span className={clsx(
                "font-mono",
                result.stationarity.pValue < 0.05 ? "text-emerald-400" : "text-amber-400"
              )}>
                {result.stationarity.pValue.toFixed(4)}
              </span>
            </div>
          </div>
        </div>

        {/* Outliers */}
        <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-4">
          <div className="flex items-center gap-3 mb-3">
            {result.outliers.percentage < 1 ? (
              <CheckCircle className="w-5 h-5 text-emerald-400" />
            ) : result.outliers.percentage < 5 ? (
              <AlertTriangle className="w-5 h-5 text-amber-400" />
            ) : (
              <XCircle className="w-5 h-5 text-red-400" />
            )}
            <h4 className="font-semibold text-slate-200">Outliers</h4>
          </div>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-slate-500">Count:</span>
              <span className="text-slate-300 font-mono">{result.outliers.count}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">Percentage:</span>
              <span className="text-slate-300 font-mono">{result.outliers.percentage.toFixed(2)}%</span>
            </div>
          </div>
        </div>

        {/* Missingness */}
        <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-4">
          <div className="flex items-center gap-3 mb-3">
            {result.missingness.percentage === 0 ? (
              <CheckCircle className="w-5 h-5 text-emerald-400" />
            ) : (
              <AlertTriangle className="w-5 h-5 text-amber-400" />
            )}
            <h4 className="font-semibold text-slate-200">Missing Values</h4>
          </div>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-slate-500">Count:</span>
              <span className="text-slate-300 font-mono">{result.missingness.count}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">Percentage:</span>
              <span className="text-slate-300 font-mono">{result.missingness.percentage.toFixed(2)}%</span>
            </div>
          </div>
        </div>

        {/* Look-ahead Bias (placeholder) */}
        <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-4">
          <div className="flex items-center gap-3 mb-3">
            <TrendingDown className="w-5 h-5 text-slate-500" />
            <h4 className="font-semibold text-slate-200">Look-ahead Bias</h4>
          </div>
          <div className="flex items-center justify-center h-12 text-sm text-slate-500">
            Analysis not yet run
          </div>
        </div>
      </div>
    </div>
  );
}
