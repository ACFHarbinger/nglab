import { useState, useEffect } from "react";
import { invoke } from "@tauri-apps/api/core";
import { X, GitCompare, TrendingUp, Cpu, Clock, HardDrive, AlertCircle } from "lucide-react";
import clsx from "clsx";

/**
 * @module components/models/ModelComparisonView
 * @description Side-by-side comparison view for selected models.
 * Displays architecture, size, and performance metrics in a comparative table.
 */

export interface ModelMetrics {
  name: string;
  architecture: string;
  size_bytes: number;
  modified_ts: number;
  // Performance metrics (optional, may be null if not yet evaluated)
  sharpe_ratio?: number;
  max_drawdown?: number;
  win_rate?: number;
  avg_reward?: number;
}

interface ModelComparisonViewProps {
  /** Names of models to compare */
  modelNames: string[];
  /** Callback to close the comparison view */
  onClose: () => void;
}

export function ModelComparisonView({ modelNames, onClose }: ModelComparisonViewProps) {
  const [models, setModels] = useState<ModelMetrics[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchModelDetails = async () => {
      setIsLoading(true);
      setError(null);
      try {
        // Fetch detailed metrics for each model
        const results = await Promise.all(
          modelNames.map((name) =>
            invoke<ModelMetrics>("get_model_details", { modelName: name })
          )
        );
        setModels(results);
      } catch (err) {
        console.error("Failed to fetch model details:", err);
        setError("Failed to load model details. Some metrics may be unavailable.");
        // Fallback: use basic metadata
        try {
          const basicList = await invoke<ModelMetrics[]>("list_trained_models");
          const filtered = basicList.filter((m) => modelNames.includes(m.name));
          setModels(filtered);
        } catch {
          setError("Unable to load model information.");
        }
      } finally {
        setIsLoading(false);
      }
    };

    if (modelNames.length > 0) {
      fetchModelDetails();
    }
  }, [modelNames]);

  const formatSize = (bytes: number) => (bytes / (1024 * 1024)).toFixed(2) + " MB";
  const formatDate = (ts: number) => new Date(ts * 1000).toLocaleDateString();
  const formatPercent = (val?: number) => (val != null ? (val * 100).toFixed(1) + "%" : "—");
  const formatNumber = (val?: number, decimals = 2) =>
    val != null ? val.toFixed(decimals) : "—";

  // Find best values for highlighting
  const getBestClass = (values: (number | undefined)[], current: number | undefined, higherIsBetter = true) => {
    const defined = values.filter((v): v is number => v != null);
    if (defined.length === 0 || current == null) return "";
    const best = higherIsBetter ? Math.max(...defined) : Math.min(...defined);
    return current === best ? "text-emerald-400 font-semibold" : "";
  };

  const metrics = [
    {
      label: "Architecture",
      icon: Cpu,
      getValue: (m: ModelMetrics) => m.architecture,
      highlight: false,
    },
    {
      label: "Size",
      icon: HardDrive,
      getValue: (m: ModelMetrics) => formatSize(m.size_bytes),
      highlight: false,
    },
    {
      label: "Modified",
      icon: Clock,
      getValue: (m: ModelMetrics) => formatDate(m.modified_ts),
      highlight: false,
    },
    {
      label: "Sharpe Ratio",
      icon: TrendingUp,
      getValue: (m: ModelMetrics) => formatNumber(m.sharpe_ratio),
      rawValue: (m: ModelMetrics) => m.sharpe_ratio,
      higherIsBetter: true,
    },
    {
      label: "Max Drawdown",
      icon: AlertCircle,
      getValue: (m: ModelMetrics) => formatPercent(m.max_drawdown),
      rawValue: (m: ModelMetrics) => m.max_drawdown,
      higherIsBetter: false,
    },
    {
      label: "Win Rate",
      icon: TrendingUp,
      getValue: (m: ModelMetrics) => formatPercent(m.win_rate),
      rawValue: (m: ModelMetrics) => m.win_rate,
      higherIsBetter: true,
    },
    {
      label: "Avg Reward",
      icon: TrendingUp,
      getValue: (m: ModelMetrics) => formatNumber(m.avg_reward, 4),
      rawValue: (m: ModelMetrics) => m.avg_reward,
      higherIsBetter: true,
    },
  ];

  return (
    <div className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm flex items-center justify-center p-6">
      <div className="bg-slate-900 border border-slate-700 rounded-2xl shadow-2xl w-full max-w-5xl max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-slate-800">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-indigo-500/20 rounded-lg">
              <GitCompare className="text-indigo-400" size={24} />
            </div>
            <div>
              <h2 className="text-xl font-bold text-white">Model Comparison</h2>
              <p className="text-sm text-slate-400">
                Comparing {modelNames.length} models
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-slate-800 rounded-lg transition-colors text-slate-400 hover:text-white"
          >
            <X size={20} />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-auto p-6">
          {isLoading ? (
            <div className="flex items-center justify-center h-64">
              <div className="animate-spin w-8 h-8 border-2 border-indigo-500 border-t-transparent rounded-full" />
            </div>
          ) : error ? (
            <div className="flex items-center justify-center h-64 text-amber-400 gap-2">
              <AlertCircle size={20} />
              <span>{error}</span>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-slate-800">
                    <th className="text-left py-3 px-4 text-slate-500 font-medium text-sm">
                      Metric
                    </th>
                    {models.map((model) => (
                      <th
                        key={model.name}
                        className="text-left py-3 px-4 text-indigo-300 font-semibold"
                      >
                        {model.name}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {metrics.map((metric) => {
                    const rawValues = metric.rawValue
                      ? models.map((m) => metric.rawValue!(m))
                      : [];

                    return (
                      <tr
                        key={metric.label}
                        className="border-b border-slate-800/50 hover:bg-slate-800/30 transition-colors"
                      >
                        <td className="py-3 px-4">
                          <div className="flex items-center gap-2 text-slate-400">
                            <metric.icon size={14} />
                            <span className="text-sm">{metric.label}</span>
                          </div>
                        </td>
                        {models.map((model) => {
                          const value = metric.getValue(model);
                          const rawVal = metric.rawValue ? metric.rawValue(model) : undefined;
                          const bestClass =
                            metric.rawValue && metric.higherIsBetter !== undefined
                              ? getBestClass(rawValues, rawVal, metric.higherIsBetter)
                              : "";

                          return (
                            <td
                              key={model.name}
                              className={clsx(
                                "py-3 px-4 text-sm font-mono",
                                bestClass || "text-slate-300"
                              )}
                            >
                              {value}
                            </td>
                          );
                        })}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-slate-800 flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg text-sm font-medium transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
