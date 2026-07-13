import {
  createChart,
  ColorType,
  IChartApi,
  ISeriesApi,
  AreaSeries,
  CandlestickSeries,
  LineSeries,
  Time,
  CandlestickData,
  WhitespaceData,
} from "lightweight-charts";
import { useEffect, useRef } from "react";
import { IndicatorConfig } from "../charts/IndicatorOverlay";
import { calculateSMA, calculateEMA, calculateBollingerBands, calculateRSI } from "../../utils/indicators";
import { ChartType } from "../charts/ChartToolbar";

/**
 * Single data point for the chart.
 * Can be simple value (Area) or OHLC (Candle).
 */
export interface ChartDataPoint {
  time: number;
  value?: number;
  open?: number;
  high?: number;
  low?: number;
  close?: number;
}

interface TerminalChartProps {
  data: ChartDataPoint[];
  color?: string;
  height?: number;
  indicators?: IndicatorConfig[];
  chartType?: ChartType;
}

export function TerminalChart({
  data,
  color = "#22c55e",
  height,
  indicators = [],
  chartType = "Area",
}: TerminalChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const mainSeriesRef = useRef<ISeriesApi<"Area"> | ISeriesApi<"Candlestick"> | null>(null);

  // Keep track of indicator series to update/remove them
  const indicatorSeriesRefs = useRef<Map<string, ISeriesApi<"Line"> | ISeriesApi<"Area">>>(new Map());

  // Early return for empty data to avoid undefined errors
  if (!data || data.length === 0) {
    return (
      <div className="w-full h-full min-h-[300px] flex items-center justify-center text-slate-500">
        Loading chart data...
      </div>
    );
  }

  // Initialize Chart
  useEffect(() => {
    if (!chartContainerRef.current) return;

    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: "transparent" },
        textColor: "#94a3b8",
      },
      grid: {
        vertLines: { color: "#1e293b", style: 1 },
        horzLines: { color: "#1e293b", style: 1 },
      },
      width: chartContainerRef.current.clientWidth,
      height: height || chartContainerRef.current.clientHeight || 400,
      timeScale: {
        timeVisible: true,
        secondsVisible: true,
        borderColor: "#1e293b",
      },
      rightPriceScale: {
        borderColor: "#1e293b",
        scaleMargins: {
          top: 0.1,
          bottom: 0.1,
        },
      },
      crosshair: {
        vertLine: {
          color: "#64748b",
          labelBackgroundColor: "#64748b",
        },
        horzLine: {
          color: "#64748b",
          labelBackgroundColor: "#64748b",
        },
      },
    });

    chartRef.current = chart;

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
  }, [height]);

  // Re-create Main Series when Chart Type changes
  useEffect(() => {
    if (!chartRef.current) return;

    // Remove old series if exists
    if (mainSeriesRef.current) {
      chartRef.current.removeSeries(mainSeriesRef.current);
      mainSeriesRef.current = null;
    }

    const chart = chartRef.current;

    if (chartType === "Area") {
      mainSeriesRef.current = chart.addSeries(AreaSeries, {
        lineColor: color,
        topColor: color + "40",
        bottomColor: color + "00",
        lineWidth: 2,
        priceFormat: { type: "price", precision: 3, minMove: 0.001 },
      });
    } else {
      // Candle or HeikinAshi (both use CandlestickSeries)
      mainSeriesRef.current = chart.addSeries(CandlestickSeries, {
        upColor: "#22c55e",
        downColor: "#f43f5e",
        borderVisible: false,
        wickUpColor: "#22c55e",
        wickDownColor: "#f43f5e",
        priceFormat: { type: "price", precision: 3, minMove: 0.001 },
      });
    }
  }, [chartType, color]);

  // Update Main Data
  useEffect(() => {
    if (!mainSeriesRef.current || !chartRef.current) return;

    // Sort data
    const sortedData = [...data].sort((a, b) => a.time - b.time);

    if (chartType === "Area") {
      // Map to Single Value
      const areaData = sortedData.map(d => ({
        time: d.time as Time,
        value: d.value ?? d.close ?? 0,
      }));
      (mainSeriesRef.current as ISeriesApi<"Area">).setData(areaData);
    } else {
      // Map to OHLC
      const candleData = sortedData.map(d => ({
        time: d.time as Time,
        open: d.open ?? d.value ?? 0,
        high: d.high ?? d.value ?? 0,
        low: d.low ?? d.value ?? 0,
        close: d.close ?? d.value ?? 0,
      }));
      (mainSeriesRef.current as ISeriesApi<"Candlestick">).setData(candleData);
    }

    // Auto fit only on init?
    if (sortedData.length > 0) {
      // chartRef.current.timeScale().fitContent();
    }
  }, [data, chartType]);

  // Manage Indicators logic remains similar but needs to use `d.close ?? d.value`
  useEffect(() => {
    if (!chartRef.current || data.length === 0) return;

    const chart = chartRef.current;
    const activeIds = new Set(indicators.map(i => i.id));
    const currentSeriesMap = indicatorSeriesRefs.current;

    // ... removal logic ...
    for (const [id, series] of currentSeriesMap.entries()) {
      if (!activeIds.has(id)) {
        chart.removeSeries(series);
        currentSeriesMap.delete(id);
      }
    }

    // Prepare price data array (use Close or Value)
    const prices = data.map(d => d.close ?? d.value ?? 0);
    const times = data.map(d => d.time as Time);

    // ... loop indicators ...
    indicators.forEach(ind => {
      let seriesData: { time: Time; value: number }[] = [];

      if (ind.type === "SMA") {
        const sma = calculateSMA(prices, ind.period);
        seriesData = sma.map((v, i) => ({ time: times[i], value: v || NaN })).filter(d => !isNaN(d.value));
      } else if (ind.type === "EMA") {
        const ema = calculateEMA(prices, ind.period);
        seriesData = ema.map((v, i) => ({ time: times[i], value: v || NaN })).filter(d => !isNaN(d.value));
      }
      // ... RSI, BB ... same as before
      else if (ind.type === "RSI") {
        const rsi = calculateRSI(prices, ind.period);
        seriesData = rsi.map((v, i) => ({ time: times[i], value: v || NaN })).filter(d => !isNaN(d.value));
        let series = currentSeriesMap.get(ind.id) as ISeriesApi<"Line">;
        if (!series) {
          series = chart.addSeries(LineSeries, {
            color: ind.color,
            lineWidth: 2,
            title: "RSI",
            priceScaleId: "rsi",
          });
          chart.priceScale("rsi").applyOptions({
            scaleMargins: { top: 0.8, bottom: 0 },
            borderVisible: false,
          });
          chart.priceScale("right").applyOptions({
            scaleMargins: { top: 0.1, bottom: 0.25 },
          });
          currentSeriesMap.set(ind.id, series);
        }
        series.setData(seriesData as any);
        return;
      } else if (ind.type === "BollingerBands") {
        const { upper, middle, lower } = calculateBollingerBands(prices, ind.period, ind.stdDev || 2);
        const updateLine = (suffix: string, values: (number | null)[], opacity: string = "") => {
          const subId = `${ind.id}-${suffix}`;
          let series = currentSeriesMap.get(subId) as ISeriesApi<"Line">;
          if (!series) {
            series = chart.addSeries(LineSeries, {
              color: ind.color + opacity,
              lineWidth: 1,
              title: `${ind.type} ${suffix}`,
              lastValueVisible: false,
              priceLineVisible: false,
            });
            currentSeriesMap.set(subId, series);
          }
          const d = values.map((v, i) => ({ time: times[i], value: v || NaN })).filter(item => !isNaN(item.value));
          series.setData(d as any);
        };
        updateLine("mid", middle);
        updateLine("upper", upper, "80");
        updateLine("lower", lower, "80");
        return;
      }

      // Default line series (SMA/EMA)
      let series = currentSeriesMap.get(ind.id) as ISeriesApi<"Line">;
      if (!series) {
        series = chart.addSeries(LineSeries, {
          color: ind.color,
          lineWidth: 2,
          title: `${ind.type} (${ind.period})`,
        });
        currentSeriesMap.set(ind.id, series);
      }
      series.setData(seriesData as any);
    });

    // RSI cleanup margin
    const hasRSI = indicators.some(i => i.type === "RSI");
    if (!hasRSI) {
      chart.priceScale("right").applyOptions({
        scaleMargins: { top: 0.1, bottom: 0.1 },
      });
    }

  }, [data, indicators]);

  return (
    <div ref={chartContainerRef} className="w-full h-full min-h-[300px]" />
  );
}
