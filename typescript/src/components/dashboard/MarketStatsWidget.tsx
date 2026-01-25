import { Activity, Users, Clock, Calendar, Zap, Eye } from "lucide-react";
import { useStreamingGuard } from "../../hooks/useStreamingGuard";
import clsx from "clsx";

/**
 * Simple SVG sparkline chart component.
 */
const Sparkline = ({
  color = "#6366f1",
  data,
}: {
  /** Line stroke color. */
  color?: string;
  /** Array of numerical values to plot. */
  data: number[];
}) => {
  const max = Math.max(...data);
  const min = Math.min(...data);
  const range = max - min || 1;
  const points = data
    .map((d, i) => {
      const x = (i / (data.length - 1)) * 100;
      const y = 100 - ((d - min) / range) * 100;
      return `${x},${y}`;
    })
    .join(" ");

  return (
    <svg
      viewBox="0 0 100 100"
      className="w-full h-12 overflow-visible"
      preserveAspectRatio="none"
    >
      <polyline
        points={points}
        fill="none"
        stroke={color}
        strokeWidth="2"
        strokeLinecap="round"
        vectorEffect="non-scaling-stroke"
      />
      <defs>
        <linearGradient
          id={`grad-${color.replace("#", "")}`}
          x1="0"
          x2="0"
          y1="0"
          y2="1"
        >
          <stop offset="0%" stopColor={color} stopOpacity="0.2" />
          <stop offset="100%" stopColor={color} stopOpacity="0" />
        </linearGradient>
      </defs>
      <polyline
        points={`0,100 ${points} 100,100`}
        fill={`url(#grad-${color.replace("#", "")})`}
        stroke="none"
      />
    </svg>
  );
};

/**
 * Widget displaying high-level market statistics.
 * Shows 24h volume, active traders, and peak activity hours with sparklines.
 */
export function MarketStatsWidget() {
  const { canStream, isLoggedIn } = useStreamingGuard();

  return (
    <div className="flex flex-col gap-4 h-full">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span className="text-sm font-bold uppercase tracking-wider text-slate-400">
            Market Overview
          </span>
          <span
            className={clsx(
              "flex items-center gap-1.5 text-[10px] font-bold px-2 py-0.5 rounded-full border transition-all duration-300",
              canStream
                ? "bg-green-500/20 text-green-400 border-green-500/30"
                : !isLoggedIn
                  ? "bg-amber-500/20 text-amber-400 border-amber-500/30"
                  : "bg-rose-500/20 text-rose-400 border-rose-500/30",
            )}
          >
            <span
              className={clsx(
                "w-1.5 h-1.5 rounded-full transition-colors duration-300",
                canStream
                  ? "bg-green-500 animate-pulse"
                  : !isLoggedIn
                    ? "bg-amber-500"
                    : "bg-rose-500",
              )}
            />
            {canStream ? "LIVE" : !isLoggedIn ? "LOGIN REQ" : "GATED"}
          </span>
        </div>
        <button className="text-[10px] text-slate-500 hover:text-white transition-colors uppercase font-bold flex items-center gap-1">
          <Activity size={10} /> Overview
        </button>
      </div>

      {/* Top Row: 3 Main Stats */}
      <div className="grid grid-cols-3 gap-4">
        {/* Card 1: Volume */}
        <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-4 flex flex-col justify-between relative overflow-hidden">
          <div className="relative z-10">
            <div className="flex items-center gap-2 text-yellow-500 mb-1">
              <Activity size={14} className="animate-pulse" />
              <span className="text-[10px] font-bold uppercase tracking-wider">
                24h Volume
              </span>
            </div>
            <div className="text-xl font-mono font-bold text-white">
              $370.158M
            </div>
          </div>
          <div className="relative z-10 mt-2">
            <Sparkline
              color="#eab308"
              data={[40, 35, 55, 45, 60, 75, 50, 65, 80, 70, 90, 85]}
            />
            <div className="flex justify-between text-[9px] text-slate-500 mt-1">
              <span>00:00</span>
              <span className="text-yellow-500">4,000,649 trades</span>
              <span>Now</span>
            </div>
          </div>
        </div>

        {/* Card 2: Active Traders */}
        <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-4 flex flex-col justify-between relative overflow-hidden">
          <div className="relative z-10">
            <div className="flex items-center gap-2 text-cyan-400 mb-1">
              <Users size={14} />
              <span className="text-[10px] font-bold uppercase tracking-wider">
                24h Active Traders
              </span>
            </div>
            <div className="text-xl font-mono font-bold text-white">
              198,510
            </div>
          </div>
          <div className="relative z-10 mt-2">
            <Sparkline
              color="#22d3ee"
              data={[20, 30, 25, 40, 35, 50, 45, 60, 55, 70, 65, 80]}
            />
            <div className="flex justify-between text-[9px] text-slate-500 mt-1">
              <span>00:00</span>
              <span className="text-cyan-400">Avg: $82.53</span>
              <span>Now</span>
            </div>
          </div>
        </div>

        {/* Card 3: Peak Hours */}
        <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-4 flex flex-col justify-between relative overflow-hidden">
          <div className="relative z-10">
            <div className="flex items-center gap-2 text-purple-400 mb-1">
              <Clock size={14} />
              <span className="text-[10px] font-bold uppercase tracking-wider">
                Peak Hours
              </span>
            </div>
            <div className="text-xl font-mono font-bold text-white">
              20:00 UTC
            </div>
          </div>
          <div className="mt-2 flex items-end gap-0.5 h-10">
            {[30, 45, 20, 60, 80, 100, 70, 50, 40, 60, 40, 30].map((h, i) => (
              <div
                key={i}
                className="flex-1 bg-purple-500/30 hover:bg-purple-500 rounded-t-sm transition-colors"
                style={{ height: `${h}%` }}
              />
            ))}
          </div>
          <div className="flex justify-between text-[9px] text-slate-500 mt-1">
            <span>12PM</span>
            <span className="text-purple-400">Peak: $29.94m</span>
            <span>11PM</span>
          </div>
        </div>
      </div>

      {/* Bottom Row: Secondary Stats */}
      <div className="grid grid-cols-4 gap-4">
        {/* Closing Today */}
        <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-3">
          <div className="flex items-center gap-2 text-slate-500 mb-1">
            <Calendar size={12} />
            <span className="text-[10px] font-bold uppercase tracking-wider">
              Closing Today
            </span>
          </div>
          <div className="flex items-baseline gap-2">
            <span className="text-lg font-mono font-bold text-rose-400">
              1721
            </span>
            <span className="text-[10px] text-slate-500">markets</span>
          </div>
        </div>

        {/* TQ Avg Volume */}
        <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-3">
          <div className="flex items-center gap-2 text-slate-500 mb-1">
            <Activity size={12} />
            <span className="text-[10px] font-bold uppercase tracking-wider">
              TQ Avg Volume
            </span>
          </div>
          <div className="text-[10px] text-slate-600 font-bold uppercase">
            Coming Soon
          </div>
        </div>

        {/* TQ Avg Traders */}
        <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-3">
          <div className="flex items-center gap-2 text-slate-500 mb-1">
            <Users size={12} />
            <span className="text-[10px] font-bold uppercase tracking-wider">
              TQ Avg Traders
            </span>
          </div>
          <div className="text-[10px] text-slate-600 font-bold uppercase">
            Coming Soon
          </div>
        </div>

        {/* Market Flow */}
        <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-3">
          <div className="flex items-center gap-2 text-slate-500 mb-1">
            <Zap size={12} />
            <span className="text-[10px] font-bold uppercase tracking-wider">
              Market Flow
            </span>
          </div>
          <div className="grid grid-cols-3 gap-2 text-center">
            <div>
              <div className="text-sm font-mono font-bold text-white">15.0</div>
              <div className="text-[8px] text-slate-500 uppercase">TPS</div>
            </div>
            <div>
              <div className="text-sm font-mono font-bold text-white">1364</div>
              <div className="text-[8px] text-slate-500 uppercase">TPM</div>
            </div>
            <div>
              <div className="text-sm font-mono font-bold text-white">
                3010.0
              </div>
              <div className="text-[8px] text-slate-500 uppercase">Max</div>
            </div>
          </div>
          <button className="mt-2 w-full bg-purple-600/20 hover:bg-purple-600/30 text-purple-400 text-[10px] font-bold py-1.5 rounded border border-purple-500/30 transition-colors flex items-center justify-center gap-1">
            <Eye size={10} /> View Intel
          </button>
        </div>
      </div>
    </div>
  );
}
