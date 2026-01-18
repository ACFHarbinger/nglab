/**
 * @module components/dashboard/FavoriteMarketsWidget
 * @description Displays user's favorited markets in a compact card layout on the dashboard.
 */
import { Star, ArrowUpRight, ChevronRight } from "lucide-react";
import clsx from "clsx";
import { FavoriteMarket } from "../../hooks/useFavorites";

interface FavoriteMarketsWidgetProps {
  favorites: FavoriteMarket[];
  livePrices?: Record<string, number>;
  onSelectMarket: (marketId: string) => void;
  onViewAll?: () => void;
}

export function FavoriteMarketsWidget({
  favorites,
  livePrices,
  onSelectMarket,
  onViewAll,
}: FavoriteMarketsWidgetProps) {
  const getPrice = (market: FavoriteMarket) => {
    if (!livePrices) return 0.5;
    if (livePrices[market.id]) return livePrices[market.id];
    if (market.marketData?.outcomes?.length) {
      const yesId = market.marketData.outcomes[0].id;
      if (livePrices[yesId]) return livePrices[yesId];
    }
    return 0.5;
  };

  const isLive = (market: FavoriteMarket) => {
    if (!livePrices) return false;
    if (livePrices[market.id]) return true;
    if (market.marketData?.outcomes?.length) {
      return livePrices[market.marketData.outcomes[0].id] !== undefined;
    }
    return false;
  };

  // Show max 4 favorites on dashboard
  const displayFavorites = favorites.slice(0, 4);

  return (
    <div className="bg-slate-900/50 border border-slate-800 rounded-xl overflow-hidden">
      {/* Header */}
      <div className="px-5 py-3 border-b border-slate-800 flex justify-between items-center bg-slate-900/50">
        <div className="flex items-center gap-2 text-yellow-500 font-bold uppercase tracking-wider text-sm">
          <Star size={16} className="fill-yellow-500" /> Your Favorites
        </div>
        {onViewAll && (
          <button
            onClick={onViewAll}
            className="flex items-center gap-1 text-[10px] text-slate-500 hover:text-white transition-colors uppercase font-bold"
          >
            Manage Favorites <ChevronRight size={12} />
          </button>
        )}
      </div>

      {/* Favorites Grid */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 p-4">
        {displayFavorites.map((market) => {
          const price = getPrice(market);
          const live = isLive(market);

          return (
            <button
              key={market.id}
              onClick={() => onSelectMarket(market.id)}
              className="group flex flex-col p-3 bg-slate-800/50 hover:bg-slate-800 border border-slate-700/50 hover:border-indigo-500/50 rounded-lg transition-all text-left"
            >
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-1.5">
                  <Star size={10} className="text-yellow-500 fill-yellow-500" />
                  <span className="text-xs font-bold text-slate-300 group-hover:text-white truncate max-w-[80px]">
                    {market.symbol}
                  </span>
                </div>
                <ArrowUpRight
                  size={12}
                  className="text-slate-600 group-hover:text-indigo-400 transition-colors"
                />
              </div>

              <p
                className="text-[10px] text-slate-500 truncate mb-2"
                title={market.name}
              >
                {market.name}
              </p>

              <div className="flex items-center gap-1.5 mt-auto">
                <span
                  className={clsx(
                    "text-sm font-mono font-bold",
                    live ? "text-emerald-400" : "text-white"
                  )}
                >
                  ${price.toFixed(3)}
                </span>
                {live && (
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
                )}
              </div>
            </button>
          );
        })}
      </div>

      {/* Show more indicator */}
      {favorites.length > 4 && (
        <div className="px-4 pb-3">
          <button
            onClick={onViewAll}
            className="w-full py-2 text-xs text-slate-500 hover:text-white bg-slate-800/30 hover:bg-slate-800/60 rounded-lg transition-all"
          >
            +{favorites.length - 4} more favorite
            {favorites.length - 4 !== 1 ? "s" : ""}
          </button>
        </div>
      )}
    </div>
  );
}
