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
} from "lucide-react";
import { invoke } from "@tauri-apps/api/core";
import {
  createChart,
  ColorType,
  IChartApi,
  UTCTimestamp,
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

type GarchResult = {
  returns: number[];
  volatility: number[];
};

type HoltWintersResult = {
  path: number[];
  used_seed?: number;
};

/**
 * Valid model types for the forecasting lab.
 */
type ActiveModelType = "arima" | "prophet" | "garch" | "holt_winters" | "trained_model";

export default function PredictionTab() {
  const [activeModel, setActiveModel] = useState<ActiveModelType>("arima");
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
  const [hwGamma] = useState(0.1);
  const [hwSeasonality] = useState(7);

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

    // @ts-expect-error - lightweight-charts type complexity
    const lineSeries = chart.addSeries('Line', {
      color: "#3b82f6",
      lineWidth: 2,
    });

    // @ts-expect-error - lightweight-charts type complexity
    const predSeries = chart.addSeries('Line', {
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

  useEffect(() => {
    if (rawData.length > 0 && selectedColumn && lineSeriesRef.current) {
      const chartData = prepareChartData(rawData, selectedColumn);
      lineSeriesRef.current.setData(chartData);
      if (chartRef.current) {
        chartRef.current.timeScale().fitContent();
      }
    }
  }, [rawData, selectedColumn]);

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

  const runPrediction = async () => {
    if (rawData.length === 0 || !selectedColumn) return;

    setIsPredicting(true);
    try {
      const dataValues = rawData
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
            times: rawData.map((r) => Number(r._ts) / 1000), // Convert to seconds
            values: dataValues,
            growth: "linear",
            seasonality_mode: "additive",
            yearly_seasonality: true,
            weekly_seasonality: true,
            daily_seasonality: false,
            seasonality_prior_scale: 10.0,
            changepoint_prior_scale: 0.05,
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
            gamma: 0.1,
            period: 7,
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
        const lastRow = rawData[rawData.length - 1];
        const lastTs = lastRow._ts;
        const lastIndex = rawData.length - 1;

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
        <button
          onClick={handleOpenFile}
          className="flex items-center gap-2 px-4 py-2 bg-slate-800 hover:bg-slate-700 rounded-lg transition-colors border border-slate-700"
        >
          <FileSpreadsheet size={18} />
          {fileName || "Load CSV Data"}
        </button>
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
                disabled={isPredicting || rawData.length === 0}
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

            {rawData.length === 0 && (
              <div className="absolute inset-0 bg-slate-950/40 backdrop-blur-[2px] flex items-center justify-center flex-col gap-2">
                <FileSpreadsheet size={48} className="text-slate-700" />
                <p className="text-slate-500 font-medium">
                  Load a CSV to start analyzing
                </p>
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
