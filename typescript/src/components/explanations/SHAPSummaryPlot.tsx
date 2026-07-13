/**
 * @module components/explanations/SHAPSummaryPlot
 * @description SHAP summary plot showing feature importance and value distribution.
 */

import { 
  ScatterChart, 
  Scatter, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  Cell
} from 'recharts';
import { Sparkles } from "lucide-react";

interface SHAPValue {
  feature: string;
  value: number;       // feature value
  shapValue: number;   // SHAP contribution
  featureIndex: number;
}

interface SHAPSummaryPlotProps {
  data: SHAPValue[] | null;
  isLoading?: boolean;
}

// Mock SHAP data for demo
export const MOCK_SHAP_DATA: SHAPValue[] = [
  // Imbalance feature
  { feature: "imbalance", value: 0.85, shapValue: 0.12, featureIndex: 0 },
  { feature: "imbalance", value: 0.62, shapValue: 0.08, featureIndex: 0 },
  { feature: "imbalance", value: -0.45, shapValue: -0.09, featureIndex: 0 },
  { feature: "imbalance", value: -0.72, shapValue: -0.11, featureIndex: 0 },
  { feature: "imbalance", value: 0.15, shapValue: 0.02, featureIndex: 0 },
  // Log Returns
  { feature: "log_ret", value: 0.02, shapValue: 0.05, featureIndex: 1 },
  { feature: "log_ret", value: -0.01, shapValue: -0.03, featureIndex: 1 },
  { feature: "log_ret", value: 0.005, shapValue: 0.01, featureIndex: 1 },
  { feature: "log_ret", value: -0.015, shapValue: -0.04, featureIndex: 1 },
  // Volatility
  { feature: "volatility", value: 0.025, shapValue: -0.02, featureIndex: 2 },
  { feature: "volatility", value: 0.018, shapValue: -0.01, featureIndex: 2 },
  { feature: "volatility", value: 0.035, shapValue: -0.04, featureIndex: 2 },
  // RSI
  { feature: "rsi", value: 72, shapValue: 0.03, featureIndex: 3 },
  { feature: "rsi", value: 35, shapValue: -0.02, featureIndex: 3 },
  { feature: "rsi", value: 55, shapValue: 0.005, featureIndex: 3 },
  // Spread
  { feature: "spread", value: 0.001, shapValue: -0.01, featureIndex: 4 },
  { feature: "spread", value: 0.002, shapValue: -0.015, featureIndex: 4 },
];

// Group by feature and calculate mean absolute SHAP
function computeFeatureImportance(data: SHAPValue[]) {
  const grouped = data.reduce((acc, d) => {
    if (!acc[d.feature]) acc[d.feature] = [];
    acc[d.feature].push(d);
    return acc;
  }, {} as Record<string, SHAPValue[]>);

  return Object.entries(grouped)
    .map(([feature, values]) => ({
      feature,
      meanAbsShap: values.reduce((sum, v) => sum + Math.abs(v.shapValue), 0) / values.length,
      values,
    }))
    .sort((a, b) => b.meanAbsShap - a.meanAbsShap);
}

function getColorForValue(value: number, min: number, max: number): string {
  // Normalize to 0-1
  const normalized = (value - min) / (max - min || 1);
  // Blue (low) to Red (high)
  const r = Math.round(normalized * 255);
  const b = Math.round((1 - normalized) * 255);
  return `rgb(${r}, 50, ${b})`;
}

export function SHAPSummaryPlot({ data, isLoading }: SHAPSummaryPlotProps) {
  if (isLoading) {
    return (
      <div className="h-64 flex items-center justify-center text-slate-500">
        <div className="animate-pulse flex items-center gap-2">
          <Sparkles size={20} />
          Computing SHAP values...
        </div>
      </div>
    );
  }

  if (!data || data.length === 0) {
    return (
      <div className="h-64 flex items-center justify-center text-slate-500">
        No SHAP data available
      </div>
    );
  }

  const featureImportance = computeFeatureImportance(data);

  return (
    <div className="space-y-6">
      {/* Summary Bar Chart - Mean |SHAP| */}
      <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-5">
        <div className="flex items-center gap-3 mb-4">
          <Sparkles size={18} className="text-purple-400" />
          <h4 className="font-semibold text-slate-200">Feature Importance (mean |SHAP|)</h4>
        </div>
        
        <div className="space-y-2">
          {featureImportance.map((feat) => {
            const maxShap = featureImportance[0].meanAbsShap;
            const widthPercent = (feat.meanAbsShap / maxShap) * 100;
            
            return (
              <div key={feat.feature} className="flex items-center gap-3">
                <span className="text-slate-400 text-sm w-24 truncate">{feat.feature}</span>
                <div className="flex-1 h-5 bg-slate-800 rounded overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-blue-500 to-red-500 rounded"
                    style={{ width: `${widthPercent}%` }}
                  />
                </div>
                <span className="text-slate-500 text-xs font-mono w-12 text-right">
                  {feat.meanAbsShap.toFixed(3)}
                </span>
              </div>
            );
          })}
        </div>
      </div>

      {/* Beeswarm-style Scatter Plot */}
      <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-5">
        <div className="flex items-center justify-between mb-4">
          <h4 className="font-semibold text-slate-200">SHAP Value Distribution</h4>
          <div className="flex items-center gap-4 text-xs">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded bg-blue-500" />
              <span className="text-slate-500">Low value</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded bg-red-500" />
              <span className="text-slate-500">High value</span>
            </div>
          </div>
        </div>

        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 10, right: 30, left: 80, bottom: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis 
                type="number" 
                dataKey="shapValue" 
                name="SHAP value" 
                stroke="#94a3b8"
                fontSize={10}
                domain={['auto', 'auto']}
              />
              <YAxis 
                type="category" 
                dataKey="feature" 
                name="Feature" 
                stroke="#94a3b8"
                fontSize={11}
                width={75}
              />
              <Tooltip 
                contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155' }}
              />
              <Scatter data={data}>
                {data.map((entry, index) => {
                  const featureValues = data.filter(d => d.feature === entry.feature);
                  const min = Math.min(...featureValues.map(d => d.value));
                  const max = Math.max(...featureValues.map(d => d.value));
                  return (
                    <Cell 
                      key={`cell-${index}`} 
                      fill={getColorForValue(entry.value, min, max)}
                      opacity={0.7}
                    />
                  );
                })}
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
