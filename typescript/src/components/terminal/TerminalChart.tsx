import {
  createChart,
  ColorType,
  IChartApi,
  ISeriesApi,
  AreaSeries,
  LineSeries,
  Time,
} from "lightweight-charts";
import { useEffect, useRef } from "react";
import { IndicatorConfig } from "../charts/IndicatorOverlay";
import { calculateSMA, calculateEMA, calculateBollingerBands, calculateRSI } from "../../utils/indicators";

/**
 * Single data point for the chart.
 */
interface ChartDataPoint {
  /** Unix timestamp in seconds. */
  time: number;
  /** Price value. */
  value: number;
}

/**
 * Props for the TerminalChart component.
 */
interface TerminalChartProps {
  /** Time-series data points. */
  data: ChartDataPoint[];
  /** Line color hex code. */
  color?: string;
  /** Explicit height in pixels. */
  height?: number;
  /** Active indicators configuration. */
  indicators?: IndicatorConfig[];
}

/**
 * Lightweight chart optimized for terminal view.
 * Renders a price area chart with 'modern crypto' styling.
 * Automatically handles container resizing.
 */
export function TerminalChart({
  data,
  color = "#22c55e",
  height,
  indicators = [],
}: TerminalChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const mainSeriesRef = useRef<ISeriesApi<"Area"> | null>(null);

  // Keep track of indicator series to update/remove them
  const indicatorSeriesRefs = useRef<Map<string, ISeriesApi<"Line"> | ISeriesApi<"Area">>>(new Map());

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

    // Use Area series for that "modern crypto" look
    const areaSeries = chart.addSeries(AreaSeries, {
      lineColor: color,
      topColor: color + "40", // 25% opacity
      bottomColor: color + "00", // 0% opacity
      lineWidth: 2,
      priceFormat: {
        type: "price",
        precision: 3,
        minMove: 0.001,
      },
    });

    chartRef.current = chart;
    mainSeriesRef.current = areaSeries;

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
  }, [color, height]);

  // Update Main Data
  useEffect(() => {
    if (!mainSeriesRef.current || !chartRef.current) return;

    // safe copy and sort
    const sortedData = [...data].sort((a, b) => a.time - b.time);
    mainSeriesRef.current.setData(sortedData as any);

    // Only auto-fit on initial load or significant changes, otherwise it jumps too much? 
    // Usually fitContent is good.
    if (sortedData.length > 0) {
      // Check if we are scrolled to end? For now just fit.
      // chartRef.current.timeScale().fitContent(); 
    }
  }, [data]);

  // Manage Indicators
  useEffect(() => {
    if (!chartRef.current || data.length === 0) return;

    const chart = chartRef.current;
    const activeIds = new Set(indicators.map(i => i.id));
    const currentSeriesMap = indicatorSeriesRefs.current;

    // Remove old indicators
    for (const [id, series] of currentSeriesMap.entries()) {
      if (!activeIds.has(id)) {
        chart.removeSeries(series);
        currentSeriesMap.delete(id);
      }
    }

    // Prepare price data array for calculations
    const prices = data.map(d => d.value);
    const times = data.map(d => d.time as Time);

    // Add/Update new indicators
    indicators.forEach(ind => {
      // Calculate data
      let seriesData: { time: Time; value: number }[] = [];
      let extraSeriesData: { time: Time; value: number }[] | null = null; // For bands

      if (ind.type === "SMA") {
        const sma = calculateSMA(prices, ind.period);
        seriesData = sma.map((v, i) => ({ time: times[i], value: v || NaN })).filter(d => !isNaN(d.value));
      } else if (ind.type === "EMA") {
        const ema = calculateEMA(prices, ind.period);
        seriesData = ema.map((v, i) => ({ time: times[i], value: v || NaN })).filter(d => !isNaN(d.value));
      } else if (ind.type === "BollingerBands") {
        const { upper, middle, lower } = calculateBollingerBands(prices, ind.period, ind.stdDev || 2);
        // We will render middle as main, and upper/lower as additional lines?
        // Actually, just rendering middle line for simplicity in this map structure is tricky.
        // Let's render middle line.
        // For bands, we ideally need multiple series per indicator config.
        // Hack: We'll construct unique IDs like "id-upper", "id-lower".

        // This loop handles the main series map. Special handling for multi-series indicators.
      } else if (ind.type === "RSI") {
        // RSI needs a separate pane/scale. Lightweight Charts supports panes by stacking charts, 
        // but single chart overlay is harder for oscillator.
        // We'll skip proper RSI for now or verify if we can add a separate scale.
        // Adding separate scale:
        // chart.addSeries(..., { priceScaleId: 'rsi-scale' });
        // chart.priceScale('rsi-scale').applyOptions({ scaleMargins: { top: 0.7, bottom: 0 } });
        const rsi = calculateRSI(prices, ind.period);
        seriesData = rsi.map((v, i) => ({ time: times[i], value: v || NaN })).filter(d => !isNaN(d.value));
      }

      // Special handling for BB
      if (ind.type === "BollingerBands") {
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
        updateLine("upper", upper, "80"); // faint
        updateLine("lower", lower, "80");
        return; // Done for BB
      }

      // Special handling for RSI (Oscillator)
      if (ind.type === "RSI") {
        let series = currentSeriesMap.get(ind.id) as ISeriesApi<"Line">;
        if (!series) {
          series = chart.addSeries(LineSeries, {
            color: ind.color,
            lineWidth: 2,
            title: "RSI",
            priceScaleId: "rsi", // Separate scale
          });
          // Configure the RSI scale to be at the bottom
          chart.priceScale("rsi").applyOptions({
            scaleMargins: {
              top: 0.8, // Take up bottom 20%
              bottom: 0,
            },
            borderVisible: false,
          });
          // Adjust main scale to not overlap
          chart.priceScale("right").applyOptions({
            scaleMargins: {
              top: 0.1,
              bottom: 0.25, // Leave room for RSI
            },
          });
          currentSeriesMap.set(ind.id, series);
        }
        series.setData(seriesData as any);
        return;
      }


      // Standard Line Indicators (SMA/EMA)
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

    // Handle removal of sub-series for BB if switched off
    // (This works via the activeIds check at start of effect)

    // Restore margins if RSI removed
    const hasRSI = indicators.some(i => i.type === "RSI");
    if (!hasRSI) {
      chart.priceScale("right").applyOptions({
        scaleMargins: { top: 0.1, bottom: 0.1 },
      });
      // We can't easily remove the 'custom' scale but it won't show if no series attached?
    }

  }, [data, indicators]);

  return (
    <div ref={chartContainerRef} className="w-full h-full min-h-[300px]" />
  );
}
