import { ArrowDownRight, Maximize2, Edit2 } from "lucide-react";
import { createChart, ColorType, AreaSeries } from "lightweight-charts";
import { useEffect, useRef } from "react";

export function UserProfileWidget() {
  const chartContainerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!chartContainerRef.current) return;

    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: "#0f172a" }, // slate-900
        textColor: "#64748b",
      },
      grid: {
        vertLines: { visible: false },
        horzLines: { visible: false },
      },
      width: chartContainerRef.current.clientWidth,
      height: 60,
      rightPriceScale: { visible: false },
      timeScale: { visible: false },
      handleScroll: false,
      handleScale: false,
      crosshair: {
        vertLine: { visible: false, labelVisible: false },
        horzLine: { visible: false, labelVisible: false },
      },
    });

    // Generate smoother dummy data
    const data = [];
    let val = 100;
    const date = new Date("2024-01-01");
    for (let i = 0; i < 50; i++) {
      val += (Math.random() - 0.5) * 5;
      date.setDate(date.getDate() + 1);
      data.push({
        time: date.toISOString().split("T")[0],
        value: val,
      });
    }

    // Determine color based on trend
    const isProfit = data[data.length - 1].value >= data[0].value;
    const lineColor = isProfit ? "#22c55e" : "#ef4444"; // green-500 : red-500
    const topColor = isProfit
      ? "rgba(34, 197, 94, 0.4)"
      : "rgba(239, 68, 68, 0.4)";
    const bottomColor = isProfit
      ? "rgba(34, 197, 94, 0.0)"
      : "rgba(239, 68, 68, 0.0)";

    const series = chart.addSeries(AreaSeries, {
      lineColor: lineColor,
      topColor: topColor,
      bottomColor: bottomColor,
      lineWidth: 2,
      crosshairMarkerVisible: false,
    });

    series.setData(data as any);
    chart.timeScale().fitContent();

    const handleResize = () => {
      if (chartContainerRef.current) {
        chart.applyOptions({ width: chartContainerRef.current.clientWidth });
        chart.timeScale().fitContent();
      }
    };

    const resizeObserver = new ResizeObserver(() => {
      handleResize();
    });

    resizeObserver.observe(chartContainerRef.current);

    return () => {
      resizeObserver.disconnect();
      chart.remove();
    };
  }, []);

  return (
    <div className="h-full flex flex-col gap-4">
      {/* Profile Header Card */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 flex flex-col gap-4">
        <div className="flex items-start justify-between">
          <div className="flex items-center gap-3">
            <div className="w-12 h-12 rounded-full bg-indigo-600/20 border border-indigo-500/50 overflow-hidden relative">
              <img
                src="https://api.dicebear.com/7.x/avataaars/svg?seed=Harbinger"
                alt="Avatar"
                className="w-full h-full object-cover"
              />
            </div>
            <div>
              <h3 className="font-bold text-slate-100 text-lg">HarbingerACF</h3>
              <div className="text-xs text-slate-500 flex items-center gap-2">
                <span>Joined Nov 2024</span>
                <span>•</span>
                <span>50 views</span>
              </div>
            </div>
          </div>
          <div className="flex gap-2">
            <button className="p-1.5 text-slate-400 hover:text-white transition-colors">
              <Maximize2 size={16} />
            </button>
            <button className="p-1.5 text-slate-400 hover:text-white transition-colors">
              <Edit2 size={16} />
            </button>
          </div>
        </div>

        <div className="flex flex-col gap-4 border-t border-slate-800 pt-4">
          <div className="flex justify-between items-center">
            <div className="text-[10px] uppercase text-slate-500 font-semibold">
              Positions Value
            </div>
            <div className="text-lg font-bold text-white">$828.93</div>
          </div>
          <div className="flex justify-between items-center">
            <div className="text-[10px] uppercase text-slate-500 font-semibold">
              Biggest Win
            </div>
            <div className="text-lg font-bold text-white">$170.40</div>
          </div>
          <div className="flex justify-between items-center">
            <div className="text-[10px] uppercase text-slate-500 font-semibold">
              Predictions
            </div>
            <div className="text-lg font-bold text-white">14</div>
          </div>
        </div>
      </div>

      {/* PnL Card */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 flex-1 flex flex-col">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            <ArrowDownRight size={16} className="text-red-500" />
            <span className="text-slate-400 text-sm font-medium">
              Profit/Loss
            </span>
          </div>
          <div className="flex bg-slate-800 rounded p-0.5">
            <button className="px-2 py-0.5 text-[10px] font-medium text-slate-400 hover:text-white">
              1D
            </button>
            <button className="px-2 py-0.5 text-[10px] font-medium bg-slate-700 text-white rounded shadow-sm">
              1W
            </button>
            <button className="px-2 py-0.5 text-[10px] font-medium text-slate-400 hover:text-white">
              1M
            </button>
            <button className="px-2 py-0.5 text-[10px] font-medium text-slate-400 hover:text-white">
              ALL
            </button>
          </div>
        </div>

        <div className="flex items-baseline gap-2 mb-1">
          <h2 className="text-3xl font-bold text-white">-$9.30</h2>
          <span className="text-slate-500 text-xs bg-slate-800 px-1.5 py-0.5 rounded-full cursor-help">
            i
          </span>
        </div>
        <div className="text-xs text-slate-500 mb-4">Past Week</div>

        <div ref={chartContainerRef} className="w-full flex-1 min-h-[60px]" />

        <div className="flex justify-end mt-2">
          <div className="text-slate-600 text-xs font-semibold flex items-center gap-1">
            <svg
              className="w-4 h-4 opacity-50"
              viewBox="0 0 24 24"
              fill="currentColor"
            >
              <path d="M12 2L2 22H22L12 2Z" />
            </svg>
            Polymarket
          </div>
        </div>
      </div>
    </div>
  );
}
