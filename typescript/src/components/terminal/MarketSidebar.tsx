import { useState, useMemo } from "react";
import {
  Search,
  TrendingUp,
  TrendingDown,
  Star,
  StarOff,
  Filter,
  ChevronDown,
  Layers,
} from "lucide-react";
import clsx from "clsx";
import { useExchange } from "../../hooks/useExchange";
import { useMarket } from "../../hooks/useMarket";


/**
 * Represents a market entity with essential trading data.
 */
export interface Market {
  /** Unique market identifier. */
  id: string;
  /** Ticker symbol (e.g., "BTC", "TRUMP"). */
  symbol: string;
  /** Full market question/title. */
  name: string;
  /** Current trading price (0.0 - 1.0). */
  price: number;
  /** 24-hour price change percentage. */
  change24h: number;
  /** 24-hour trading volume in USD. */
  volume24h: number;
  /** Whether the market is in user's favorites. */
  isFavorite?: boolean;
  /** Supplemental market data (Polymarket specific). */
  marketData?: {
    id: string;
    outcomes?: Array<{ id: string; name: string }>;
  };
}

/**
 * Props for the MarketSidebar component.
 */
interface MarketSidebarProps {
  /** List of available markets. */
  markets: Market[];
  /** ID of the currently selected market. */
  activeMarketId?: string;
  /** Callback when a market is selected. */
  onSelectMarket: (id: string) => void;
  /** Real-time price map. */
  livePrices?: Record<string, number>;
  /** Set of favorite market IDs. */
  favoriteIds?: Set<string>;
  /** Callback to toggle favorite status. */
  onToggleFavorite?: (market: Market) => void;
}

type FilterMode = "all" | "favorites" | "multi-outcome";

/**
 * Sidebar component for navigating between different markets.
 * specific features include search, filtering by category/favorites,
 * and live price updates in the list view.
 */
export function MarketSidebar({
  markets,
  activeMarketId,
  onSelectMarket,
  livePrices,
  favoriteIds,
  onToggleFavorite,
}: MarketSidebarProps) {
  const {
    exchanges,
    activeExchange,
    setActiveExchange,
    loading: loadingExchanges,
  } = useExchange();
  const { searchResults, isSearching, searchMarkets } = useMarket();

  const [searchQuery, setSearchQuery] = useState("");
  const [filterMode, setFilterMode] = useState<FilterMode>("all");
  const [showFilterMenu, setShowFilterMenu] = useState(false);

  // Filter and search markets
  const filteredMarkets = useMemo(() => {
    let result = markets;

    // Apply search filter
    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase();
      result = result.filter(
        (m) =>
          m.name.toLowerCase().includes(query) ||
          m.symbol.toLowerCase().includes(query),
      );
    }

    // Apply category filter
    switch (filterMode) {
      case "favorites":
        result = result.filter((m) => favoriteIds?.has(m.id));
        break;
      case "multi-outcome":
        result = result.filter(
          (m) => (m.marketData?.outcomes?.length ?? 2) > 2,
        );
        break;
    }

    return result;
  }, [markets, searchQuery, filterMode, favoriteIds]);

  const filterLabels: Record<FilterMode, string> = {
    all: "All Markets",
    favorites: "Favorites",
    "multi-outcome": "Multi-Outcome",
  };

  const getOutcomeCount = (market: Market) => {
    return market.marketData?.outcomes?.length ?? 2;
  };

  const isFavorited = (market: Market) => {
    return favoriteIds?.has(market.id) || false;
  };

  const renderMarketItem = (market: Market) => {
    // Resolve price: check livePrices by markeId OR by outcome ID (Polymarket)
    let currentPrice = market.price;
    let isLive = false;

    if (livePrices) {
      if (livePrices[market.id] !== undefined) {
        currentPrice = livePrices[market.id];
        isLive = true;
      } else if (
        market.marketData?.outcomes &&
        market.marketData.outcomes.length > 0
      ) {
        const yesId = market.marketData.outcomes[0].id;
        if (livePrices[yesId] !== undefined) {
          currentPrice = livePrices[yesId];
          isLive = true;
        }
      }
    }

    const outcomeCount = getOutcomeCount(market);
    const favorited = isFavorited(market);

    return (
      <div
        key={market.id}
        className={clsx(
          "group px-3 py-3 border-b border-slate-800/50 cursor-pointer transition-colors hover:bg-slate-800/50",
          activeMarketId === market.id
            ? "bg-slate-800 border-l-2 border-l-indigo-500"
            : "border-l-2 border-l-transparent",
        )}
      >
        <div className="flex justify-between items-start mb-1">
          <div
            className="flex items-center gap-2 flex-1"
            onClick={() => onSelectMarket(market.id)}
          >
            <span
              className={clsx(
                "font-bold text-sm",
                activeMarketId === market.id
                  ? "text-white"
                  : "text-slate-300 group-hover:text-white",
              )}
            >
              {market.symbol}
            </span>
            {outcomeCount > 2 && (
              <span className="flex items-center gap-0.5 text-[10px] px-1.5 py-0.5 bg-indigo-500/20 text-indigo-400 rounded-full">
                <Layers size={8} />
                {outcomeCount}
              </span>
            )}
            {isLive && (
              <span className="flex h-1.5 w-1.5 rounded-full bg-emerald-500 animate-pulse" />
            )}
          </div>

          <div className="flex items-center gap-2">
            <span
              className={clsx(
                "font-mono text-sm",
                isLive ? "text-emerald-400" : "text-white",
              )}
              onClick={() => onSelectMarket(market.id)}
            >
              {isFinite(currentPrice)
                ? currentPrice < 1
                  ? currentPrice.toFixed(3)
                  : currentPrice.toFixed(2)
                : "0.000"}
            </span>

            {/* Favorite Toggle Button */}
            {onToggleFavorite && (
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onToggleFavorite(market);
                }}
                className={clsx(
                  "p-1 rounded transition-colors",
                  favorited
                    ? "text-yellow-500 hover:text-yellow-400"
                    : "text-slate-600 hover:text-yellow-500 opacity-0 group-hover:opacity-100",
                )}
                title={favorited ? "Remove from favorites" : "Add to favorites"}
              >
                {favorited ? (
                  <Star className="h-3.5 w-3.5 fill-current" />
                ) : (
                  <StarOff className="h-3.5 w-3.5" />
                )}
              </button>
            )}
          </div>
        </div>

        <div
          className="flex justify-between items-center text-xs"
          onClick={() => onSelectMarket(market.id)}
        >
          <span
            className="text-slate-500 truncate max-w-[120px]"
            title={market.name}
          >
            {market.name}
          </span>
          <div className="flex items-center gap-1">
            <span className="text-slate-600">
              $
              {isFinite(market.volume24h)
                ? (market.volume24h / 1000).toFixed(1)
                : "0.0"}
              k
            </span>
            <span
              className={clsx(
                "flex items-center",
                market.change24h >= 0 ? "text-green-400" : "text-rose-400",
              )}
            >
              {market.change24h >= 0 ? (
                <TrendingUp size={10} className="mr-0.5" />
              ) : (
                <TrendingDown size={10} className="mr-0.5" />
              )}
              {Math.abs(market.change24h || 0).toFixed(2)}%
            </span>
          </div>
        </div>

        {/* Show outcomes preview for multi-outcome markets */}
        {outcomeCount > 2 && market.marketData?.outcomes && (
          <div className="mt-2 flex flex-wrap gap-1">
            {market.marketData.outcomes.slice(0, 4).map((outcome) => (
              <span
                key={outcome.id}
                className="text-[9px] px-1.5 py-0.5 bg-slate-800 text-slate-400 rounded"
              >
                {outcome.name}
              </span>
            ))}
            {outcomeCount > 4 && (
              <span className="text-[9px] px-1.5 py-0.5 text-slate-600">
                +{outcomeCount - 4} more
              </span>
            )}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="flex flex-col h-full bg-slate-900 border-r border-slate-800 w-80">
      {/* Header / Search */}
      <div className="p-3 border-b border-slate-800 space-y-3">
        {/* Exchange Selector */}
        <div className="relative">
          <label className="text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-1.5 block">
            Exchange Source
          </label>
          <div className="flex items-center gap-2">
            <select
              value={activeExchange}
              onChange={(e) => setActiveExchange(e.target.value)}
              className="flex-1 bg-slate-950 border border-slate-800 rounded-lg px-3 py-1.5 text-xs text-slate-200 focus:outline-none focus:ring-1 focus:ring-indigo-500 appearance-none cursor-pointer"
            >
              {exchanges.map((ex) => (
                <option key={ex} value={ex}>
                  {ex}
                </option>
              ))}
            </select>
            <div className="absolute right-3 top-[26px] pointer-events-none">
              <ChevronDown size={12} className="text-slate-500" />
            </div>
            {loadingExchanges ? (
              <span className="w-1.5 h-1.5 rounded-full bg-indigo-500 animate-pulse" />
            ) : (
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.5)]" title="Connected" />
            )}
          </div>
        </div>

        <div className="relative">
          <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-slate-500" />
          <input
            type="text"
            placeholder={`Search ${activeExchange} markets...`}
            value={searchQuery}
            onChange={(e) => {
              setSearchQuery(e.target.value);
              // In a real app we'd debounce this
              if (e.target.value.length > 2) {
                searchMarkets(e.target.value);
              }
            }}
            className="w-full bg-slate-950 border border-slate-800 rounded-lg pl-9 pr-4 py-2 text-sm text-slate-200 placeholder:text-slate-600 focus:outline-none focus:ring-1 focus:ring-indigo-500"
          />
          {isSearching && (
            <div className="absolute right-3 top-2.5">
              <div className="w-4 h-4 border-2 border-indigo-500/20 border-t-indigo-500 rounded-full animate-spin" />
            </div>
          )}
        </div>

        {/* Filter Dropdown */}
        <div className="relative">
          <button
            onClick={() => setShowFilterMenu(!showFilterMenu)}
            className="flex items-center justify-between w-full px-3 py-1.5 bg-slate-800 hover:bg-slate-700 rounded-lg text-xs text-slate-300 transition-colors"
          >
            <span className="flex items-center gap-2">
              <Filter size={12} />
              {filterLabels[filterMode]}
            </span>
            <ChevronDown
              size={12}
              className={clsx(
                "transition-transform",
                showFilterMenu && "rotate-180",
              )}
            />
          </button>

          {showFilterMenu && (
            <div className="absolute z-10 top-full left-0 right-0 mt-1 bg-slate-800 border border-slate-700 rounded-lg shadow-xl overflow-hidden">
              {(Object.keys(filterLabels) as FilterMode[]).map((mode) => (
                <button
                  key={mode}
                  onClick={() => {
                    setFilterMode(mode);
                    setShowFilterMenu(false);
                  }}
                  className={clsx(
                    "w-full px-3 py-2 text-left text-xs transition-colors",
                    filterMode === mode
                      ? "bg-indigo-600 text-white"
                      : "text-slate-300 hover:bg-slate-700",
                  )}
                >
                  {filterLabels[mode]}
                  {mode === "favorites" && favoriteIds && (
                    <span className="ml-2 text-slate-500">
                      ({favoriteIds.size})
                    </span>
                  )}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Market List */}
      <div className="flex-1 overflow-y-auto custom-scrollbar">
        {/* Search Results if searching */}
        {searchQuery.length > 2 && (
          <div className="py-2 animate-in fade-in duration-300">
            <div className="px-3 py-1 flex items-center justify-between text-[10px] font-bold text-indigo-400 uppercase tracking-widest bg-indigo-500/5 mb-1">
              <span>{activeExchange} Results</span>
              {isSearching && <span className="text-[9px] animate-pulse">Searching...</span>}
            </div>
            {searchResults.length === 0 && !isSearching ? (
              <div className="px-4 py-8 text-center text-xs text-slate-600">
                No results found for "{searchQuery}"
              </div>
            ) : (
              searchResults.map((r) => renderMarketItem({
                id: r.id,
                symbol: r.id.substring(0, 10).toUpperCase(),
                name: r.title,
                price: 0,
                change24h: 0,
                volume24h: r.volume || 0,
                marketData: { id: r.id, outcomes: r.outcomes.map(o => ({ id: o, name: o })) }
              }))
            )}
            <div className="h-px bg-slate-800 my-4 mx-3" />
          </div>
        )}

        {filteredMarkets.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-slate-500 p-4">
            <Search size={32} className="mb-2 opacity-50" />
            <p className="text-sm text-center">
              {filterMode === "favorites"
                ? "No favorite markets yet"
                : filterMode === "multi-outcome"
                  ? "No multi-outcome markets found"
                  : "No markets match your search"}
            </p>
          </div>
        ) : (
          <>
            {/* Favorites Section (Pinned at top if in 'all' mode) */}
            {filterMode === "all" &&
              Array.from(favoriteIds || []).length > 0 && (
                <div className="py-2">
                  <div className="px-3 py-1 flex items-center gap-2 text-[10px] font-bold text-slate-500 uppercase tracking-widest bg-slate-800/20 mb-1">
                    <Star
                      size={10}
                      className="fill-yellow-500 text-yellow-500"
                    />
                    Favorites
                  </div>
                  {filteredMarkets
                    .filter((m) => isFavorited(m))
                    .map((market) => renderMarketItem(market))}
                </div>
              )}

            {/* Main Section */}
            <div>
              {filterMode === "all" &&
                Array.from(favoriteIds || []).length > 0 && (
                  <div className="px-3 py-1 flex items-center gap-2 text-[10px] font-bold text-slate-500 uppercase tracking-widest bg-slate-800/20 mb-1">
                    <Layers size={10} />
                    Other Markets
                  </div>
                )}
              {filteredMarkets
                .filter((m) => filterMode !== "all" || !isFavorited(m))
                .map((market) => renderMarketItem(market))}
            </div>
          </>
        )}
      </div>

      {/* Footer Stats */}
      <div className="p-2 border-t border-slate-800 text-xs text-slate-500 flex justify-between">
        <span>{filteredMarkets.length} markets</span>
        {favoriteIds && favoriteIds.size > 0 && (
          <span className="flex items-center gap-1">
            <Star size={10} className="fill-yellow-500 text-yellow-500" />
            {favoriteIds.size} favorites
          </span>
        )}
      </div>
    </div>
  );
}
