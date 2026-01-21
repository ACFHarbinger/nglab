/**
 * @module components/PredictionTab
 * @description Advanced forecasting lab integrating econometric and deep learning models.
 */
import { useState, useEffect, useRef } from "react";
import {
  Brain,
  Loader2,
  LineChart as LineChartIcon,
  Activity,
  FileSpreadsheet,
  Wifi,
} from "lucide-react";
import { invoke } from "@tauri-apps/api/core";
import {
  createChart,
  ColorType,
  IChartApi,
  UTCTimestamp,
  LineSeries,
} from "lightweight-charts";
import { open } from "@tauri-apps/plugin-dialog";
import { readTextFile } from "@tauri-apps/plugin-fs";
import Papa from "papaparse";
import { prepareChartData } from "../utils/dataHelpers";
import { ArimaResult } from "../../../rust/bindings/ArimaResult";
import { ProphetResult } from "../../../rust/bindings/ProphetResult";
import {
  ArimaParamsSchema,
  ProphetParamsSchema,
  GarchParamsSchema,
  HoltWintersParamsSchema,
} from "../validation";

/**
 * Interface representing a row from a CSV file.
 */
interface CsvRow {
  [key: string]: string | number | null | undefined;
  /** Internal timestamp field calculated during processing. */
  _ts?: number;
}

/**
 * Result structure from a GARCH volatility model execution.
 */
type GarchResult = {
  /** Array of historical returns. */
  returns: number[];
  /** Array of conditional volatility estimates. */
  volatility: number[];
};

/**
 * Result structure from a Holt-Winters exponential smoothing model.
 */
type HoltWintersResult = {
  /** Forecasted path values. */
  path: number[];
  /** Random seed used (if applicable). */
  used_seed?: number;
};

import { MarketMetadata } from "../hooks/usePolymarket";

/**
 * Valid model types for the forecasting lab.
 */
type ActiveModelType = "arima" | "prophet" | "garch" | "holt_winters" | "trained_model";

/**
 * Props for the PredictionTab component.
 */
interface PredictionTabProps {
  /** Map of current live prices from the active stream. */
  livePrices: Record<string, number>;
  /** Whether the market stream is currently active. */
  isStreaming: boolean;
  /** Metadata for the currently selected market. */
  activeMarket: MarketMetadata | null;
}

export default function PredictionTab({
  livePrices,
  isStreaming,
  activeMarket,
}: PredictionTabProps) {
  const [activeModel, setActiveModel] = useState<ActiveModelType>("arima");
  const [dataSource, setDataSource] = useState<"csv" | "live">("csv");
  const [liveHistory, setLiveHistory] = useState<CsvRow[]>([]);
  const latestPricesRef = useRef(livePrices);

  // Keep ref updated for polling
  useEffect(() => {
    latestPricesRef.current = livePrices;
  }, [livePrices]);
  const [isPredicting, setIsPredicting] = useState(false);
  const [rawData, setRawData] = useState<CsvRow[]>([]);
  const [columns, setColumns] = useState<string[]>([]);
  const [selectedColumn, setSelectedColumn] = useState<string>("");
  const [fileName, setFileName] = useState<string>("");

  // Model Parameters
  const [arimaP, setArimaP] = useState(1);
  const [arimaD, setArimaD] = useState(1);
  const [arimaQ, setArimaQ] = useState(1);
  const [steps, setSteps] = useState(10);
  const [hwAlpha, setHwAlpha] = useState(0.2);
  const [hwBeta, setHwBeta] = useState(0.1);
  const [hwGamma, setHwGamma] = useState(0.1);
  const [hwSeasonality, setHwSeasonality] = useState(7);

  // Prophet Parameters
  const [prophetGrowth, setProphetGrowth] = useState<"linear" | "flat">("linear");
  const [seasonalityMode, setSeasonalityMode] = useState<"additive" | "multiplicative">("additive");
  const [yearlySeasonality, setYearlySeasonality] = useState(true);
  const [weeklySeasonality, setWeeklySeasonality] = useState(true);
  const [dailySeasonality, setDailySeasonality] = useState(false);

  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const lineSeriesRef = useRef<any>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const predictionSeriesRef = useRef<any>(null);

  useEffect(() => {
    if (!chartContainerRef.current) return;

    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: "transparent" },
        textColor: "#94a3b8",
      },
      grid: {
        vertLines: { color: "#334155" },
        horzLines: { color: "#334155" },
      },
      width: chartContainerRef.current.clientWidth,
      height: 400,
    });

    const lineSeries = chart.addSeries(LineSeries, {
      color: "#3b82f6",
      lineWidth: 2,
    });

    const predSeries = chart.addSeries(LineSeries, {
      color: "#10b981",
      lineWidth: 2,
      lineStyle: 2, // Dashed
    });

    chartRef.current = chart;
    lineSeriesRef.current = lineSeries;
    predictionSeriesRef.current = predSeries;

    const handleResize = () => {
      if (chartContainerRef.current) {
        chart.applyOptions({ width: chartContainerRef.current.clientWidth });
      }
    };

    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
      chart.remove();
    };
  }, []);

  // Live Data Polling (Simplified version of AnalysisTab's logic)
  useEffect(() => {
    if (dataSource === "live" && isStreaming && activeMarket) {
      const interval = setInterval(() => {
        const now = Date.now();
        const newRow: CsvRow = { _ts: now, "Timestamp (UTC)": now };
        const currentPrices = latestPricesRef.current;
        let hasData = false;

        activeMarket.outcomes.forEach((outcome) => {
          if (currentPrices[outcome.id] !== undefined) {
            newRow[outcome.name] = currentPrices[outcome.id];
            hasData = true;
          }
        });

        if (hasData) {
          setLiveHistory((prev) => {
            const next = [...prev, newRow];
            return next.length > 500 ? next.slice(next.length - 500) : next;
          });
        }
      }, 1000);
      return () => clearInterval(interval);
    }
  }, [dataSource, isStreaming, activeMarket]);

  // Unified Data Source Selection
  const currentData = dataSource === "csv" ? rawData : liveHistory;

  // Update columns and chart when source or data changes
  useEffect(() => {
    if (dataSource === "live" && activeMarket) {
      const names = activeMarket.outcomes.map((o) => o.name);
      setColumns(names);
      if (!selectedColumn || !names.includes(selectedColumn)) {
        setSelectedColumn(names[0] || "");
      }
    } else if (dataSource === "csv" && rawData.length > 0) {
      const firstRowKeys = Object.keys(rawData[0]);
      const dateKey = firstRowKeys.find(
        (k) =>
          k.toLowerCase().includes("date") || k.toLowerCase().includes("time"),
      );
      const cols = firstRowKeys.filter(
        (k) =>
          k !== dateKey && k !== "_ts" && k !== "Timestamp (UTC)",
      );
      setColumns(cols);
    }
  }, [dataSource, activeMarket, rawData]);

  // Update Chart with unified data
  useEffect(() => {
    if (currentData.length > 0 && selectedColumn && lineSeriesRef.current) {
      const chartData = prepareChartData(currentData, selectedColumn);
      lineSeriesRef.current.setData(chartData.data);
      if (chartRef.current && dataSource === "csv") {
        chartRef.current.timeScale().fitContent();
      }
    }
  }, [currentData, selectedColumn, dataSource]);

  /**
   * Opens a native file dialog to load a CSV dataset.
   * Parses the file using PapaParse and detects date/time columns heuristically.
   * Supports EU (DD/MM/YYYY) and US (MM/DD/YYYY) date formats.
   */
  const handleOpenFile = async () => {
    try {
      const selected = await open({
        multiple: false,
        filters: [{ name: "CSV", extensions: ["csv"] }],
      });

      if (selected && typeof selected === 'string') {
        const path = selected;
        setFileName(path.split(/[/\\]/).pop() || path);
        const content = await readTextFile(path);

        Papa.parse<CsvRow>(content, {
          header: true,
          dynamicTyping: true,
          skipEmptyLines: true,
          complete: (results) => {
            const raw = results.data;
            if (raw.length > 0) {
              const firstRowKeys = Object.keys(raw[0]);
              const dateKey = firstRowKeys.find(
                (k) =>
                  k.toLowerCase().includes("date") ||
                  k.toLowerCase().includes("time") ||
                  k.toLowerCase() === "timestamp",
              );

              // Heuristic Detection of Date Format
              let isEU = false; // DD/MM/YYYY
              let isUS = false; // MM/DD/YYYY

              if (dateKey) {
                for (const row of raw) {
                  const val = row[dateKey];
                  if (typeof val === "string") {
                    const cleanVal = val.trim().replace("T", " ");
                    const datePart = cleanVal.split(" ")[0];
                    // FIX: Hyphen moved to end to resolve no-useless-escape error
                    const parts = datePart.split(/[/ -]/);
                    if (parts.length === 3) {
                      const p0 = parseInt(parts[0]);
                      const p1 = parseInt(parts[1]);

                      if (!isNaN(p0) && !isNaN(p1)) {
                        if (p0 > 12) isEU = true;
                        if (p1 > 12) isUS = true;
                      }
                    }
                  }
                  if (isEU || isUS) break;
                }
              }

              let processed = raw.map((row) => {
                let timestamp: number = NaN;

                if (dateKey && row[dateKey]) {
                  const val = row[dateKey];
                  if (typeof val === "number") {
                    timestamp = val < 10000000000 ? val * 1000 : val;
                  } else {
                    const dStr = String(val).trim().replace("T", " ");
                    const [datePart, ...timeParts] = dStr.split(" ");
                    const timePart = timeParts.join(" ");
                    // FIX: Hyphen moved to end
                    const parts = datePart.split(/[/ -]/);
                    if (parts.length === 3) {
                      if (parts[0].length === 4) {
                        timestamp = new Date(dStr).getTime();
                      } else {
                        let dateString = datePart;
                        if (isEU) {
                          dateString = `${parts[2]}/${parts[1]}/${parts[0]}`;
                        } else if (isUS) {
                          dateString = `${parts[2]}/${parts[0]}/${parts[1]}`;
                        }

                        if (timePart) {
                          dateString += " " + timePart;
                        }
                        timestamp = new Date(dateString).getTime();
                      }
                    } else {
                      timestamp = new Date(dStr).getTime();
                    }

                    if (isNaN(timestamp)) {
                      timestamp = new Date(dStr).getTime();
                    }
                  }
                }
                return { ...row, _ts: timestamp };
              });

              const validDateCount = processed.filter(
                (r) => r._ts !== undefined && !isNaN(r._ts),
              ).length;

              if (validDateCount > 0) {
                processed = processed.filter(
                  (r) => r._ts !== undefined && !isNaN(r._ts),
                );
                processed.sort((a, b) => (a._ts ?? 0) - (b._ts ?? 0));
              } else {
                processed = processed.map((r) => ({ ...r, _ts: NaN }));
              }

              setRawData(processed);
              const cols = firstRowKeys.filter(
                (k) => k !== dateKey && k !== "_ts",
              );
              setColumns(cols);
              const defaultCol =
                cols.find(
                  (c) =>
                    c.toLowerCase().includes("price") ||
                    c.toLowerCase().includes("close"),
                ) || cols[0];
              setSelectedColumn(defaultCol);
            }
          },
        });
      }
    } catch {
      // FIX: Removed unused 'err' to satisfy linter
      console.error("Failed to open or parse CSV file");
    }
  };

  /**
   * Executes the selected forecasting model (ARIMA, Prophet, GARCH, etc.).
   * Validates parameters using Zod schemas before invoking the backend.
   * Updates the chart with the returned prediction path.
   */
  const runPrediction = async () => {
    if (currentData.length === 0 || !selectedColumn) return;

    setIsPredicting(true);
    try {
      const dataValues = currentData
        .map((r) => Number(r[selectedColumn]))
        .filter((v) => !isNaN(v));

      let result: number[] = [];

      switch (activeModel) {
        case "arima": {
          const params = {
            data: dataValues,
            p: arimaP,
            d: arimaD,
            q: arimaQ,
            steps,
          };
          ArimaParamsSchema.parse(params);
          const res = await invoke<ArimaResult>("run_arima", params);
          result = res.path;
          break;
        }
        case "prophet": {
          const params = {
            times: rawData.map((r) => Math.round(Number(r._ts) / 1000)), // Convert to seconds
            values: dataValues,
            growth: prophetGrowth,
            seasonality_mode: seasonalityMode,
            yearly_seasonality: yearlySeasonality,
            weekly_seasonality: weeklySeasonality,
            daily_seasonality: dailySeasonality,
            changepoint_prior_scale: 0.05,
            seasonality_prior_scale: 10.0,
            forecast_horizon: steps,
          };

          ProphetParamsSchema.parse(params);
          const res = await invoke<ProphetResult>("run_prophet", { params });
          result = res.values;
          break;
        }
        case "garch": {
          const params = {
            omega: 0.01,
            alpha: [0.1],
            beta: [0.8],
            steps,
            data: dataValues,
          };
          GarchParamsSchema.parse(params);
          const res = await invoke<GarchResult>("run_garch", { params });
          const lastPrice = dataValues[dataValues.length - 1];
          result = res.volatility.map((v) => lastPrice * (1 + v));
          break;
        }
        case "holt_winters": {
          const params = {
            alpha: hwAlpha,
            beta: hwBeta,
            gamma: hwGamma,
            period: hwSeasonality,
            seasonal_type: "Additive" as const,
            steps,
            sigma: 0.01,
            data: dataValues,
          };
          HoltWintersParamsSchema.parse(params);
          const res = await invoke<HoltWintersResult>("run_holt_winters", { params });
          result = res.path;
          break;
        }
        case "trained_model": {
          const res = await invoke<number[]>("run_trained_model", {
            data: dataValues,
            steps,
          });
          result = res;
          break;
        }
      }



      if (predictionSeriesRef.current && result.length > 0) {
        const lastRow = currentData[currentData.length - 1];
        const lastTs = lastRow._ts;
        const lastIndex = currentData.length - 1;

        const predData = result.map((val, i) => {
          if (lastTs && !isNaN(lastTs)) {
            return {
              time: (lastTs / 1000 + (i + 1) * 86400) as UTCTimestamp,
              value: val,
            };
          } else {
            return {
              time: (lastIndex + i + 1) as UTCTimestamp,
              value: val,
            };
          }
        });

        predictionSeriesRef.current.setData(predData);
      }
    } catch {
      // FIX: Removed unused 'error' to satisfy linter
      console.error("Prediction failed");
    } finally {
      setIsPredicting(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold flex items-center gap-2">
            <Brain className="text-blue-500" />
            Forecasting Lab
          </h2>
          <p className="text-slate-400">
            Combine traditional econometrics with modern deep learning.
          </p>
        </div>
        <div className="flex items-center gap-3">
          <div className="flex bg-slate-900 rounded-lg p-1 border border-slate-800">
            <button
              onClick={() => setDataSource("csv")}
              className={`px-3 py-1.5 text-xs font-medium rounded-md transition-all ${dataSource === "csv"
                ? "bg-slate-800 text-white shadow-sm"
                : "text-slate-500 hover:text-slate-300"
                }`}
            >
              CSV File
            </button>
            <button
              onClick={() => setDataSource("live")}
              className={`px-3 py-1.5 text-xs font-medium rounded-md transition-all ${dataSource === "live"
                ? "bg-slate-800 text-white shadow-sm"
                : "text-slate-500 hover:text-slate-300"
                }`}
            >
              Live Market
            </button>
          </div>
          {dataSource === "csv" && (
            <button
              onClick={handleOpenFile}
              className="flex items-center gap-2 px-4 py-2 bg-slate-800 hover:bg-slate-700 rounded-lg transition-colors border border-slate-700"
            >
              <FileSpreadsheet size={18} />
              {fileName || "Load CSV Data"}
            </button>
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        <div className="lg:col-span-1 space-y-6">
          <div className="bg-slate-900/50 p-4 rounded-xl border border-slate-800 space-y-4">
            <h3 className="font-semibold flex items-center gap-2 text-sm text-slate-300 uppercase tracking-wider">
              <Activity size={16} />
              Model Configuration
            </h3>

            <div className="space-y-2">
              <label className="text-xs text-slate-500">Target Column</label>
              <select
                value={selectedColumn}
                onChange={(e) => setSelectedColumn(e.target.value)}
                className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="">Select column...</option>
                {columns.map((col) => (
                  <option key={col} value={col}>
                    {col}
                  </option>
                ))}
              </select>
            </div>

            <div className="space-y-2">
              <label className="text-xs text-slate-500">Algorithm</label>
              <select
                value={activeModel}
                onChange={(e) =>
                  setActiveModel(e.target.value as ActiveModelType)
                }
                className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="arima">ARIMA (Econometric)</option>
                <option value="prophet">Prophet (Meta)</option>
                <option value="holt_winters">Holt-Winters</option>
                <option value="garch">GARCH (Volatility)</option>
                <option value="trained_model">Trained Neural Net</option>
              </select>
            </div>

            <div className="pt-4 border-t border-slate-800 space-y-4">
              {activeModel === "arima" && (
                <div className="grid grid-cols-3 gap-2">
                  <div className="space-y-1">
                    <label className="text-[10px] text-slate-500 uppercase">
                      p
                    </label>
                    <input
                      type="number"
                      value={arimaP}
                      onChange={(e) => setArimaP(parseInt(e.target.value))}
                      className="w-full bg-slate-950 border border-slate-800 rounded px-2 py-1 text-sm"
                    />
                  </div>
                  <div className="space-y-1">
                    <label className="text-[10px] text-slate-500 uppercase">
                      d
                    </label>
                    <input
                      type="number"
                      value={arimaD}
                      onChange={(e) => setArimaD(parseInt(e.target.value))}
                      className="w-full bg-slate-950 border border-slate-800 rounded px-2 py-1 text-sm"
                    />
                  </div>
                  <div className="space-y-1">
                    <label className="text-[10px] text-slate-500 uppercase">
                      q
                    </label>
                    <input
                      type="number"
                      value={arimaQ}
                      onChange={(e) => setArimaQ(parseInt(e.target.value))}
                      className="w-full bg-slate-950 border border-slate-800 rounded px-2 py-1 text-sm"
                    />
                  </div>
                </div>
              )}

              {activeModel === "prophet" && (
                <div className="space-y-3">
                  <div className="space-y-1">
                    <label className="text-[10px] text-slate-500 uppercase">Growth</label>
                    <select
                      value={prophetGrowth}
                      onChange={(e) => setProphetGrowth(e.target.value as "linear" | "flat")}
                      className="w-full bg-slate-950 border border-slate-800 rounded px-2 py-1 text-sm"
                    >
                      <option value="linear">Linear</option>
                      <option value="flat">Flat</option>
                    </select>
                  </div>
                  <div className="space-y-1">
                    <label className="text-[10px] text-slate-500 uppercase">Seasonality Mode</label>
                    <select
                      value={seasonalityMode}
                      onChange={(e) => setSeasonalityMode(e.target.value as "additive" | "multiplicative")}
                      className="w-full bg-slate-950 border border-slate-800 rounded px-2 py-1 text-sm"
                    >
                      <option value="additive">Additive</option>
                      <option value="multiplicative">Multiplicative</option>
                    </select>
                  </div>
                  <div className="flex flex-col gap-2 pt-2">
                    <label className="flex items-center gap-2 text-sm text-slate-400 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={yearlySeasonality}
                        onChange={(e) => setYearlySeasonality(e.target.checked)}
                        className="rounded bg-slate-950 border-slate-800 text-blue-600 focus:ring-blue-500"
                      />
                      Yearly Seasonality
                    </label>
                    <label className="flex items-center gap-2 text-sm text-slate-400 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={weeklySeasonality}
                        onChange={(e) => setWeeklySeasonality(e.target.checked)}
                        className="rounded bg-slate-950 border-slate-800 text-blue-600 focus:ring-blue-500"
                      />
                      Weekly Seasonality
                    </label>
                    <label className="flex items-center gap-2 text-sm text-slate-400 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={dailySeasonality}
                        onChange={(e) => setDailySeasonality(e.target.checked)}
                        className="rounded bg-slate-950 border-slate-800 text-blue-600 focus:ring-blue-500"
                      />
                      Daily Seasonality
                    </label>
                  </div>
                </div>
              )}

              {activeModel === "holt_winters" && (
                <div className="space-y-3">
                  <div className="space-y-1">
                    <div className="flex justify-between text-[10px] text-slate-500 uppercase">
                      <span>Alpha (Level)</span>
                      <span>{hwAlpha}</span>
                    </div>
                    <input
                      type="range"
                      min="0"
                      max="1"
                      step="0.01"
                      value={hwAlpha}
                      onChange={(e) => setHwAlpha(parseFloat(e.target.value))}
                      className="w-full"
                    />
                  </div>
                  <div className="space-y-1">
                    <div className="flex justify-between text-[10px] text-slate-500 uppercase">
                      <span>Beta (Trend)</span>
                      <span>{hwBeta}</span>
                    </div>
                    <input
                      type="range"
                      min="0"
                      max="1"
                      step="0.01"
                      value={hwBeta}
                      onChange={(e) => setHwBeta(parseFloat(e.target.value))}
                      className="w-full"
                    />
                  </div>
                  <div className="space-y-1">
                    <div className="flex justify-between text-[10px] text-slate-500 uppercase">
                      <span>Gamma (Seasonality)</span>
                      <span>{hwGamma}</span>
                    </div>
                    <input
                      type="range"
                      min="0"
                      max="1"
                      step="0.01"
                      value={hwGamma}
                      onChange={(e) => setHwGamma(parseFloat(e.target.value))}
                      className="w-full"
                    />
                  </div>
                  <div className="space-y-1">
                    <label className="text-[10px] text-slate-500 uppercase">
                      Seasonality Period
                    </label>
                    <input
                      type="number"
                      value={hwSeasonality}
                      onChange={(e) => setHwSeasonality(parseInt(e.target.value))}
                      className="w-full bg-slate-950 border border-slate-800 rounded px-2 py-1 text-sm"
                    />
                  </div>
                </div>
              )}

              <div className="space-y-1">
                <label className="text-[10px] text-slate-500 uppercase">
                  Steps to Forecast
                </label>
                <input
                  type="number"
                  value={steps}
                  onChange={(e) => setSteps(parseInt(e.target.value))}
                  className="w-full bg-slate-950 border border-slate-800 rounded px-2 py-1 text-sm"
                />
              </div>

              <button
                onClick={runPrediction}
                disabled={isPredicting || currentData.length === 0}
                className="w-full py-2 bg-blue-600 hover:bg-blue-500 disabled:bg-slate-800 disabled:text-slate-500 text-white rounded-lg font-medium transition-all flex items-center justify-center gap-2"
              >
                {isPredicting ? (
                  <Loader2 size={16} className="animate-spin" />
                ) : (
                  <Brain size={16} />
                )}
                Run Prediction
              </button>
            </div>
          </div>
        </div>

        <div className="lg:col-span-3 space-y-6">
          <div className="bg-slate-900/50 p-6 rounded-2xl border border-slate-800 relative overflow-hidden">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-blue-500/10 rounded-lg">
                  <LineChartIcon size={20} className="text-blue-500" />
                </div>
                <h3 className="font-semibold">Time Series Projection</h3>
              </div>
              <div className="flex items-center gap-4 text-xs">
                <div className="flex items-center gap-1.5">
                  <div className="w-3 h-0.5 bg-blue-500" />
                  <span className="text-slate-400">Historical</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <div className="w-3 h-0.5 border-t-2 border-dashed border-emerald-500" />
                  <span className="text-slate-400">Forecast</span>
                </div>
              </div>
            </div>

            <div ref={chartContainerRef} className="w-full" />

            {currentData.length === 0 && (
              <div className="absolute inset-0 bg-slate-950/40 backdrop-blur-[2px] flex items-center justify-center flex-col gap-2">
                {dataSource === "csv" ? (
                  <>
                    <FileSpreadsheet size={48} className="text-slate-700" />
                    <p className="text-slate-500 font-medium">
                      Load a CSV to start analyzing
                    </p>
                  </>
                ) : (
                  <>
                    <Wifi size={48} className={`text-slate-700 ${isStreaming ? 'animate-pulse text-indigo-500/50' : ''}`} />
                    <p className="text-slate-500 font-medium text-center px-6">
                      {isStreaming
                        ? `Waiting for live data from ${activeMarket?.title || 'market'}...`
                        : "Start a market stream in the Markets tab to see live data here"}
                    </p>
                  </>
                )}
              </div>
            )}
          </div>

          <div className="bg-slate-900/50 p-6 rounded-2xl border border-slate-800">
            <h4 className="text-sm font-semibold text-slate-300 mb-3 uppercase tracking-wider">
              About the {activeModel.toUpperCase()} Model
            </h4>
            <div className="text-sm text-slate-400 leading-relaxed">
              {activeModel === "arima" ? (
                <p>
                  ARIMA (AutoRegressive Integrated Moving Average) is a
                  statistical model for analyzing and forecasting time series
                  data. It uses historical values and error terms to predict
                  future points, assuming the series can be made stationary
                  through differencing (d).
                </p>
              ) : activeModel === "garch" ? (
                <p>
                  GARCH models use historical returns to estimate initial
                  conditional variance. This allows for more realistic volatility
                  forecasting based on recent market regimes found in your data.
                </p>
              ) : activeModel === "holt_winters" ? (
                <p>
                  Holt-Winters (Triple Exponential Smoothing) captures level,
                  trend, and seasonality. Adjust Alpha, Beta, and Gamma to control
                  how much weight is given to recent vs. old data for each
                  component.
                </p>
              ) : activeModel === "trained_model" ? (
                <p>
                  Uses a pre-trained PyTorch model from your{" "}
                  <code>model_weights</code> directory. The model is loaded in a
                  separate Python process to run inference on the recent context
                  window.
                </p>
              ) : (
                <p>
                  Prophet uses a decomposable time series model with three main
                  components: trend, seasonality, and holidays. It is robust to
                  missing data and shifts in the trend, and specifically designed
                  for business forecasting.
                </p>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
