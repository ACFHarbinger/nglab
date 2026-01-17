/**
 * @module components/dashboard/TrendingMarketsWidget
 * @description Visualizes the most active and hot markets with probability bars and sparklines.
 */
/**
 * @module components/dashboard/UserProfileWidget
 * @description Displays user account summary, equity curves, and performance metrics.
 */
import { TrendingUp, Flame } from "lucide-react";
import clsx from "clsx";

interface TrendingMarketsProps {
  onSelectMarket: (id: string) => void;
}

// Mock Data for the list - updated to match pve.trade style
const TRENDING_DATA = [
  {
    id: "1",
    name: "Fed decision in January?",
    vol: "$43.905M",
    yesPrice: 0.0,
    noPrice: 1.0,
    change: 0.02,
    outcomes: 4,
    type: "MONTHLY",
    yesPercent: 0,
    trend: [10, 20, 15, 10, 12, 10, 8, 5, 2, 0],
    icon: "🏛️",
  },
  {
    id: "2",
    name: "Who will Trump nominate as Fed Chair?",
    vol: "$11.797M",
    yesPrice: 0.14,
    noPrice: 0.85,
    change: 0.1,
    outcomes: 39,
    type: "MONTHLY",
    yesPercent: 14,
    trend: [20, 22, 25, 30, 45, 50, 55, 60, 80, 95],
    icon: "🇺🇸",
  },
  {
    id: "3",
    name: "US strikes Iran by...?",
    vol: "$13.609M",
    yesPrice: 0.08,
    noPrice: 0.93,
    change: 0.02,
    outcomes: 14,
    type: "MONTHLY",
    yesPercent: 8,
    trend: [10, 12, 11, 15, 20, 18, 25, 30, 40, 35],
    icon: "🇺🇸",
  },
  {
    id: "4",
    name: "Suns vs. Pistons",
    vol: "$5.404M",
    yesPrice: 1.0,
    noPrice: 0.0,
    change: 0.02,
    outcomes: 48,
    type: "DAILY",
    yesPercent: 100,
    trend: [50, 50, 50, 55, 50, 60, 80, 90, 95, 100],
    icon: "🏀",
  },
  {
    id: "5",
    name: "XRP Up or Down - January 17, 5:45AM-6:...",
    vol: "$1.00",
    yesPrice: 1.0,
    noPrice: 0.0,
    change: 0.0,
    outcomes: 2,
    type: "DAILY",
    yesPercent: 100,
    trend: [40, 45, 42, 50, 55, 60, 58, 65, 70, 75],
    icon: "💰",
  },
  {
    id: "6",
    name: "Ethereum Up or Down - January 17, 5:45...",
    vol: "$3.25",
    yesPrice: 0.65,
    noPrice: 0.35,
    change: 0.0,
    outcomes: 2,
    type: "DAILY",
    yesPercent: 65,
    trend: [30, 35, 40, 45, 50, 55, 60, 65, 60, 65],
    icon: "💎",
  },
  {
    id: "7",
    name: "Will the price of XRP be between $2.10...",
    vol: "$40.00",
    yesPrice: 0.99,
    noPrice: 0.01,
    change: 0.0,
    outcomes: 2,
    type: "DAILY",
    yesPercent: 99,
    trend: [90, 92, 94, 95, 96, 97, 98, 99, 99, 99],
    icon: "📊",
  },
];

const MiniSpark = ({ data, color }: { data: number[]; color: string }) => (
  <svg viewBox="0 0 60 24" className="w-20 h-8 overflow-visible">
    <polyline
      points={data
        .map((d, i) => `${(i / (data.length - 1)) * 60},${24 - (d / 100) * 24}`)
        .join(" ")}
      fill="none"
      stroke={color}
      strokeWidth="2"
      strokeLinecap="round"
    />
  </svg>
);

const ProbabilityBar = ({ yesPercent }: { yesPercent: number }) => (
  <div className="flex items-center gap-1 text-[9px] font-mono mt-1">
    <span className="text-emerald-400">Y</span>
    <div className="flex-1 h-0.5 bg-slate-700 rounded-full overflow-hidden relative">
      <div
        className="absolute left-0 top-0 h-full bg-gradient-to-r from-emerald-500 to-emerald-400"
        style={{ width: `${yesPercent}%` }}
      />
      <div
        className="absolute right-0 top-0 h-full bg-gradient-to-l from-rose-500 to-rose-400"
        style={{ width: `${100 - yesPercent}%` }}
      />
    </div>
    <span className="text-rose-400">N</span>
    <span className="text-slate-500 ml-1">{yesPercent}%</span>
  </div>
);

export function TrendingMarketsWidget({
  onSelectMarket,
}: TrendingMarketsProps) {
  return (
    <div className="flex flex-col h-full bg-slate-900/50 border border-slate-800 rounded-xl overflow-hidden">
      <div className="px-5 py-4 border-b border-slate-800 flex justify-between items-center bg-slate-900/50">
        <div className="flex items-center gap-2 text-rose-500 font-bold uppercase tracking-wider text-sm">
          <Flame size={16} className="text-orange-500" /> Trending Markets
        </div>
        <div className="flex items-center gap-4">
          <span className="text-[10px] text-slate-500 uppercase font-bold">
            Top by Volume
          </span>
          <button className="text-[10px] text-slate-500 hover:text-white transition-colors uppercase font-bold">
            View All Markets +
          </button>
        </div>
      </div>

      <div className="flex-1 overflow-auto">
        <table className="w-full text-left border-collapse">
          <thead>
            <tr className="text-[10px] text-slate-500 uppercase font-bold border-b border-slate-800/50">
              <th className="px-5 py-3 font-medium">Market</th>
              <th className="px-5 py-3 font-medium text-center">Chart</th>
              <th className="px-5 py-3 font-medium text-right">Price</th>
              <th className="px-5 py-3 font-medium text-right">24h</th>
              <th className="px-5 py-3 font-medium text-right">Volume</th>
              <th className="px-5 py-3 font-medium text-right">Outcomes</th>
            </tr>
          </thead>
          <tbody className="text-sm font-mono text-slate-300">
            {TRENDING_DATA.map((market) => (
              <tr
                key={market.id}
                onClick={() => onSelectMarket(market.id)}
                className="group cursor-pointer hover:bg-slate-800/40 border-b border-slate-800/30 transition-colors"
              >
                <td className="px-5 py-3">
                  <div className="flex items-center gap-3">
                    <div className="w-9 h-9 rounded-full bg-slate-800 flex items-center justify-center border border-slate-700 group-hover:border-indigo-500/50 transition-colors text-lg">
                      {market.icon}
                    </div>
                    <div className="flex flex-col min-w-0">
                      <div className="flex items-center gap-2">
                        <TrendingUp
                          size={12}
                          className="text-orange-500 shrink-0"
                        />
                        <span
                          className={clsx(
                            "text-[9px] font-bold px-1.5 py-0.5 rounded uppercase",
                            market.type === "MONTHLY"
                              ? "bg-indigo-500/20 text-indigo-400"
                              : "bg-slate-700 text-slate-400",
                          )}
                        >
                          {market.type}
                        </span>
                      </div>
                      <span className="font-sans font-medium text-slate-200 group-hover:text-white transition-colors truncate max-w-[240px] text-sm mt-0.5">
                        {market.name}
                      </span>
                      <ProbabilityBar yesPercent={market.yesPercent} />
                    </div>
                  </div>
                </td>
                <td className="px-5 py-3 text-center">
                  <MiniSpark
                    data={market.trend}
                    color={market.change >= 0 ? "#10b981" : "#f43f5e"}
                  />
                </td>
                <td className="px-5 py-3 text-right">
                  <div className="font-bold text-white">
                    ${market.yesPrice.toFixed(2)}
                  </div>
                  <div className="text-[10px] text-slate-500">
                    ${market.noPrice.toFixed(2)} NO
                  </div>
                </td>
                <td className="px-5 py-3 text-right">
                  <span
                    className={clsx(
                      "text-xs",
                      market.change > 0
                        ? "text-emerald-400"
                        : market.change < 0
                          ? "text-rose-400"
                          : "text-slate-500",
                    )}
                  >
                    {market.change > 0 ? "↑" : market.change < 0 ? "↓" : "~"}{" "}
                    {Math.abs(market.change).toFixed(1)}%
                  </span>
                </td>
                <td className="px-5 py-3 text-right text-slate-400">
                  {market.vol}
                </td>
                <td className="px-5 py-3 text-right">
                  <span className="text-indigo-400 font-bold">
                    {market.outcomes}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
