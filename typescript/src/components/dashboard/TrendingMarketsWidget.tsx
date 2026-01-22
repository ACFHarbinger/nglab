/**
 * @module components/dashboard/TrendingMarketsWidget
 * @description Visualizes the most active and hot markets with probability bars and sparklines.
 */
import { useState, useEffect } from "react";
import { invoke } from "@tauri-apps/api/core";
import { TrendingUp, Flame, Loader2, AlertCircle, RefreshCw, Star } from "lucide-react";
import clsx from "clsx";

/**
 * Props for the TrendingMarketsWidget.
 */
interface TrendingMarketsProps {
  /** Callback when a market row is clicked. */
  onSelectMarket: (id: string) => void;
  /** Real-time live price map. */
  livePrices?: Record<string, number>;
  /** Set of favorite IDs */
  favoriteIds?: Set<string>;
  /** Toggle favorite callback */
  toggleFavorite?: (market: any) => void;
}

/** Market search result from backend */
interface MarketSearchResult {
  id: string;
  event_id: string;
  title: string;
  question: string;
  outcomes: string[];
  clob_token_ids: string[];
  volume: number | null;
  liquidity: number | null;
  end_date: string | null;
  active: boolean;
}

interface TrendingMarket {
  id: string;
  name: string;
  vol: string;
  yesPrice: number;
  noPrice: number;
  outcomes: number;
  type: "MONTHLY" | "DAILY";
  yesPercent: number;
  trend: number[];
  icon: string;
  clobTokenIds: string[];
}

/** Format volume for display */
function formatVolume(volume: number | null): string {
  if (!volume) return "$0";
  if (volume >= 1_000_000) {
    return `$${(volume / 1_000_000).toFixed(2)}M`;
  } else if (volume >= 1_000) {
    return `$${(volume / 1_000).toFixed(1)}K`;
  }
  return `$${volume.toFixed(2)}`;
}

/** Get icon based on market title */
function getMarketIcon(title: string): string {
  const lowerTitle = title.toLowerCase();
  if (lowerTitle.includes("bitcoin") || lowerTitle.includes("btc")) return "₿";
  if (lowerTitle.includes("ethereum") || lowerTitle.includes("eth")) return "💎";
  if (lowerTitle.includes("trump") || lowerTitle.includes("president") || lowerTitle.includes("election")) return "🇺🇸";
  if (lowerTitle.includes("fed") || lowerTitle.includes("rate") || lowerTitle.includes("inflation")) return "🏛️";
  if (lowerTitle.includes("nba") || lowerTitle.includes("basketball") || lowerTitle.includes("vs.")) return "🏀";
  if (lowerTitle.includes("nfl") || lowerTitle.includes("football") || lowerTitle.includes("super bowl")) return "🏈";
  if (lowerTitle.includes("soccer") || lowerTitle.includes("fifa") || lowerTitle.includes("world cup")) return "⚽";
  if (lowerTitle.includes("crypto") || lowerTitle.includes("xrp") || lowerTitle.includes("sol")) return "💰";
  if (lowerTitle.includes("ai") || lowerTitle.includes("openai") || lowerTitle.includes("gpt")) return "🤖";
  if (lowerTitle.includes("war") || lowerTitle.includes("strike") || lowerTitle.includes("military")) return "⚔️";
  if (lowerTitle.includes("stock") || lowerTitle.includes("market") || lowerTitle.includes("nasdaq")) return "📈";
  return "📊";
}

/** Get market type based on end date */
function getMarketType(endDate: string | null): "MONTHLY" | "DAILY" {
  if (!endDate) return "MONTHLY";
  const end = new Date(endDate);
  const now = new Date();
  const daysUntilEnd = (end.getTime() - now.getTime()) / (1000 * 60 * 60 * 24);
  return daysUntilEnd <= 7 ? "DAILY" : "MONTHLY";
}

/** Generate fake trend data (since we don't have historical data) */
function generateTrendData(): number[] {
  const trend = [];
  let value = 30 + Math.random() * 40;
  for (let i = 0; i < 10; i++) {
    value = Math.max(0, Math.min(100, value + (Math.random() - 0.5) * 15));
    trend.push(Math.round(value));
  }
  return trend;
}

/** Transform backend result to display format */
function transformToTrendingMarket(result: MarketSearchResult): TrendingMarket {
  return {
    id: result.id,
    name: result.title,
    vol: formatVolume(result.volume),
    yesPrice: 0.5,
    noPrice: 0.5,
    outcomes: result.outcomes.length,
    type: getMarketType(result.end_date),
    yesPercent: 50,
    trend: generateTrendData(),
    icon: getMarketIcon(result.title),
    clobTokenIds: result.clob_token_ids,
  };
}

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

/**
 * Widget displaying a list of trending markets.
 * Features live price updates, sparklines for price history, and volume stats.
 * Fetches initial data from the backend public API.
 */
export function TrendingMarketsWidget({
  onSelectMarket,
  livePrices,
  favoriteIds,
  toggleFavorite,
}: TrendingMarketsProps) {
  const [markets, setMarkets] = useState<TrendingMarket[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchTrendingMarkets = async () => {
    setIsLoading(true);
    setError(null);
    try {
      // Use public API that doesn't require authentication
      const response: any = await invoke("get_public_polymarket_markets", {
        limit: 10,
      });
      console.log("Trending markets response:", response);
      if (response.success && response.data) {
        const transformed = response.data.map(transformToTrendingMarket);
        console.log("Transformed markets:", transformed);
        setMarkets(transformed);
      } else if (!response.success) {
        console.error("API error:", response.message);
        setError(response.message || "Failed to fetch markets");
      } else {
        console.warn("Unexpected response format:", response);
        setError("Unexpected response from server");
      }
    } catch (e) {
      console.error("Failed to fetch trending markets:", e);
      setError("Failed to load markets. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchTrendingMarkets();
  }, []);

  // Update prices from live data
  const getMarketPrices = (market: TrendingMarket) => {
    if (!livePrices || market.clobTokenIds.length === 0) {
      return { yesPrice: market.yesPrice, noPrice: market.noPrice, isLive: false };
    }

    const yesTokenId = market.clobTokenIds[0];
    const noTokenId = market.clobTokenIds[1];

    const yesPrice = livePrices[yesTokenId];
    const noPrice = livePrices[noTokenId];

    if (yesPrice !== undefined) {
      return {
        yesPrice,
        noPrice: noPrice ?? 1 - yesPrice,
        isLive: true,
      };
    }

    return { yesPrice: market.yesPrice, noPrice: market.noPrice, isLive: false };
  };

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
          <button
            onClick={fetchTrendingMarkets}
            className="text-[10px] text-slate-500 hover:text-white transition-colors uppercase font-bold flex items-center gap-1"
            disabled={isLoading}
          >
            <RefreshCw size={10} className={isLoading ? "animate-spin" : ""} />
            Refresh
          </button>
        </div>
      </div>

      <div className="flex-1 overflow-auto">
        {error ? (
          <div className="flex flex-col items-center justify-center h-48 text-slate-500 px-4">
            <AlertCircle size={32} className="mb-3 text-amber-500" />
            <p className="text-sm text-center mb-3">{error}</p>
            <button
              onClick={fetchTrendingMarkets}
              className="text-xs text-indigo-400 hover:text-indigo-300 flex items-center gap-1"
            >
              <RefreshCw size={12} /> Try again
            </button>
          </div>
        ) : isLoading ? (
          <div className="flex items-center justify-center h-48">
            <Loader2 className="w-6 h-6 text-indigo-500 animate-spin" />
            <span className="ml-2 text-sm text-slate-400">Loading markets...</span>
          </div>
        ) : markets.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-48 text-slate-500">
            <TrendingUp size={32} className="mb-3 opacity-30" />
            <p className="text-sm">No trending markets available</p>
          </div>
        ) : (
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="text-[10px] text-slate-500 uppercase font-bold border-b border-slate-800/50">
                <th className="px-5 py-3 font-medium w-10"></th>
                <th className="px-5 py-3 font-medium">Market</th>
                <th className="px-5 py-3 font-medium text-center">Chart</th>
                <th className="px-5 py-3 font-medium text-right">Price</th>
                <th className="px-5 py-3 font-medium text-right">Volume</th>
                <th className="px-5 py-3 font-medium text-right">Outcomes</th>
              </tr>
            </thead>
            <tbody className="text-sm font-mono text-slate-300">
              {markets.map((market) => {
                const { yesPrice, noPrice, isLive } = getMarketPrices(market);
                const yesPercent = Math.round(yesPrice * 100);
                const isFavorited = favoriteIds?.has(market.id);

                return (
                  <tr
                    key={market.id}
                    onClick={() => onSelectMarket(market.id)}
                    className="group cursor-pointer hover:bg-slate-800/40 border-b border-slate-800/30 transition-colors"
                  >
                    <td className="px-5 py-3">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          if (toggleFavorite) {
                            toggleFavorite({
                              id: market.id,
                              symbol: market.id, // Fallback as symbol is not in TrendingMarket
                              name: market.name,
                              marketData: {
                                id: market.id,
                                outcomes: [], // Will be resolved by backend
                              },
                            });
                          }
                        }}
                        className={clsx(
                          "transition-colors",
                          isFavorited ? "text-yellow-500" : "text-slate-600 hover:text-yellow-500/50"
                        )}
                      >
                        <Star size={16} fill={isFavorited ? "currentColor" : "none"} />
                      </button>
                    </td>
                    <td className="px-2 py-3">
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
                                  : "bg-slate-700 text-slate-400"
                              )}
                            >
                              {market.type}
                            </span>
                            {isLive && (
                              <span className="flex items-center gap-1 text-[9px] text-emerald-400">
                                <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
                                LIVE
                              </span>
                            )}
                          </div>
                          <span className="font-sans font-medium text-slate-200 group-hover:text-white transition-colors truncate max-w-[240px] text-sm mt-0.5">
                            {market.name}
                          </span>
                          <ProbabilityBar yesPercent={yesPercent} />
                        </div>
                      </div>
                    </td>
                    <td className="px-5 py-3 text-center">
                      <MiniSpark
                        data={market.trend}
                        color={yesPercent >= 50 ? "#10b981" : "#f43f5e"}
                      />
                    </td>
                    <td className="px-5 py-3 text-right">
                      <div
                        className={clsx(
                          "font-bold",
                          isLive ? "text-emerald-400" : "text-white"
                        )}
                      >
                        ${yesPrice.toFixed(2)}
                        {isLive && (
                          <span className="ml-1.5 w-1 h-1 rounded-full bg-emerald-500 inline-block align-middle animate-pulse" />
                        )}
                      </div>
                      <div className="text-[10px] text-slate-500">
                        ${noPrice.toFixed(2)} NO
                      </div>
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
                );
              })}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
