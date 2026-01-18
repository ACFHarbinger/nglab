import { useState, useEffect, useRef, useCallback } from "react";
import { invoke } from "@tauri-apps/api/core";
import {
  Search,
  Star,
  X,
  Loader2,
  ArrowUpRight,
  AlertCircle,
} from "lucide-react";
import clsx from "clsx";
import { FavoriteMarket } from "../hooks/useFavorites";

interface FavoritesTabProps {
  favorites: FavoriteMarket[];
  favoriteIds: Set<string>;
  addFavorite: (market: Omit<FavoriteMarket, "addedAt">) => void;
  removeFavorite: (marketId: string) => void;
  isFavorite: (marketId: string) => boolean;
  toggleFavorite: (market: Omit<FavoriteMarket, "addedAt">) => void;
  livePrices?: Record<string, number>;
  onNavigateToMarket?: (marketId: string) => void;
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

interface SearchableMarket {
  id: string;
  symbol: string;
  name: string;
  price: number;
  volume: number;
  marketData?: {
    id: string;
    outcomes?: Array<{ id: string; name: string }>;
  };
}

/** Transform backend search result to frontend market format */
function transformSearchResult(result: MarketSearchResult): SearchableMarket {
  return {
    id: result.id,
    symbol: result.title.substring(0, 12).toUpperCase().replace(/\s+/g, ""),
    name: result.title,
    price: 0.5,
    volume: result.volume || 0,
    marketData: {
      id: result.id,
      outcomes: result.outcomes.map((name, i) => ({
        id: result.clob_token_ids[i] || "",
        name,
      })),
    },
  };
}

export function FavoritesTab({
  favorites,
  addFavorite,
  removeFavorite,
  isFavorite,
  livePrices,
  onNavigateToMarket,
}: FavoritesTabProps) {
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState<SearchableMarket[]>([]);
  const [openMarkets, setOpenMarkets] = useState<SearchableMarket[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isSearching, setIsSearching] = useState(false);
  const [hasSearched, setHasSearched] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const searchTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Fetch open markets on mount
  useEffect(() => {
    fetchOpenMarkets();
  }, []);

  const fetchOpenMarkets = async () => {
    setIsLoading(true);
    setError(null);
    try {
      // Use public API that doesn't require authentication
      const response: any = await invoke("get_public_polymarket_markets", {
        limit: 30,
      });
      if (response.success && response.data) {
        const markets = response.data.map(transformSearchResult);
        setOpenMarkets(markets);
      } else if (!response.success) {
        setError(response.message || "Failed to fetch markets");
      }
    } catch (e) {
      console.error("Failed to fetch open markets:", e);
      setError("Failed to connect to Polymarket. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  // Debounced search with backend API (public, no auth required)
  const searchMarkets = useCallback(async (query: string) => {
    if (query.trim().length < 2) {
      setSearchResults([]);
      setHasSearched(false);
      return;
    }

    setIsSearching(true);
    setHasSearched(true);
    setError(null);

    try {
      const response: any = await invoke("search_public_polymarket_markets", {
        query: query.trim(),
        limit: 30,
      });

      if (response.success && response.data) {
        const markets = response.data.map(transformSearchResult);
        setSearchResults(markets);
      } else if (!response.success) {
        setError(response.message || "Search failed");
        setSearchResults([]);
      }
    } catch (e) {
      console.error("Failed to search markets:", e);
      setError("Search failed. Please try again.");
      setSearchResults([]);
    } finally {
      setIsSearching(false);
    }
  }, []);

  // Debounce search input
  useEffect(() => {
    if (searchTimeoutRef.current) {
      clearTimeout(searchTimeoutRef.current);
    }

    if (searchQuery.trim() === "") {
      setSearchResults([]);
      setHasSearched(false);
      return;
    }

    searchTimeoutRef.current = setTimeout(() => {
      searchMarkets(searchQuery);
    }, 300);

    return () => {
      if (searchTimeoutRef.current) {
        clearTimeout(searchTimeoutRef.current);
      }
    };
  }, [searchQuery, searchMarkets]);

  const getMarketPrice = (market: SearchableMarket | FavoriteMarket) => {
    if (!livePrices) return "price" in market ? market.price : 0.5;

    if (livePrices[market.id]) return livePrices[market.id];

    if (market.marketData?.outcomes?.length) {
      const yesId = market.marketData.outcomes[0].id;
      if (livePrices[yesId]) return livePrices[yesId];
    }

    return "price" in market ? market.price : 0.5;
  };

  return (
    <div className="h-full flex flex-col bg-slate-950 overflow-hidden">
      {/* Header */}
      <div className="px-6 py-4 border-b border-slate-800">
        <h2 className="text-xl font-bold text-white mb-1">Favorite Markets</h2>
        <p className="text-sm text-slate-400">
          Search and add markets to your favorites for quick access
        </p>
      </div>

      <div className="flex-1 flex overflow-hidden">
        {/* Left Panel: Search & Add */}
        <div className="w-1/2 border-r border-slate-800 flex flex-col">
          {/* Search Input */}
          <div className="p-4 border-b border-slate-800">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search markets to add..."
                className="w-full bg-slate-900 border border-slate-700 rounded-lg pl-10 pr-4 py-2.5 text-sm text-slate-200 placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
              />
              {searchQuery && (
                <button
                  onClick={() => setSearchQuery("")}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-500 hover:text-slate-300"
                >
                  <X size={16} />
                </button>
              )}
            </div>
          </div>

          {/* Search Results or Suggestions */}
          <div className="flex-1 overflow-y-auto custom-scrollbar">
            {error ? (
              <div className="flex flex-col items-center justify-center h-32 text-slate-500 px-4">
                <AlertCircle size={24} className="mb-2 text-amber-500" />
                <p className="text-sm text-center">{error}</p>
                <button
                  onClick={fetchOpenMarkets}
                  className="mt-3 text-xs text-indigo-400 hover:text-indigo-300"
                >
                  Try again
                </button>
              </div>
            ) : isLoading ? (
              <div className="flex items-center justify-center h-32">
                <Loader2 className="w-6 h-6 text-indigo-500 animate-spin" />
              </div>
            ) : hasSearched ? (
              isSearching ? (
                <div className="flex items-center justify-center h-32">
                  <Loader2 className="w-5 h-5 text-indigo-500 animate-spin" />
                  <span className="ml-2 text-sm text-slate-400">
                    Searching...
                  </span>
                </div>
              ) : searchResults.length > 0 ? (
                <div className="divide-y divide-slate-800/50">
                  {searchResults.map((market) => (
                    <MarketSearchItem
                      key={market.id}
                      market={market}
                      isFavorite={isFavorite(market.id)}
                      onToggle={() => {
                        if (isFavorite(market.id)) {
                          removeFavorite(market.id);
                        } else {
                          addFavorite({
                            id: market.id,
                            symbol: market.symbol,
                            name: market.name,
                            marketData: market.marketData,
                          });
                        }
                      }}
                      price={getMarketPrice(market)}
                    />
                  ))}
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center h-32 text-slate-500">
                  <Search size={24} className="mb-2 opacity-50" />
                  <p className="text-sm">No markets found for "{searchQuery}"</p>
                  <p className="text-xs text-slate-600 mt-1">
                    Try a different search term
                  </p>
                </div>
              )
            ) : (
              <div className="p-4">
                <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3">
                  Open Markets
                </h3>
                <div className="divide-y divide-slate-800/50">
                  {openMarkets.map((market) => (
                    <MarketSearchItem
                      key={market.id}
                      market={market}
                      isFavorite={isFavorite(market.id)}
                      onToggle={() => {
                        if (isFavorite(market.id)) {
                          removeFavorite(market.id);
                        } else {
                          addFavorite({
                            id: market.id,
                            symbol: market.symbol,
                            name: market.name,
                            marketData: market.marketData,
                          });
                        }
                      }}
                      price={getMarketPrice(market)}
                    />
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Right Panel: Current Favorites */}
        <div className="w-1/2 flex flex-col">
          <div className="p-4 border-b border-slate-800">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-semibold text-slate-300">
                Your Favorites
              </h3>
              <span className="text-xs text-slate-500 bg-slate-800 px-2 py-0.5 rounded-full">
                {favorites.length} market{favorites.length !== 1 ? "s" : ""}
              </span>
            </div>
          </div>

          <div className="flex-1 overflow-y-auto custom-scrollbar">
            {favorites.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-full text-slate-500 px-4">
                <Star size={32} className="mb-3 opacity-30" />
                <p className="text-sm text-center">
                  No favorites yet. Search for markets on the left to add them.
                </p>
              </div>
            ) : (
              <div className="divide-y divide-slate-800/50">
                {favorites.map((market) => (
                  <FavoriteMarketItem
                    key={market.id}
                    market={market}
                    price={getMarketPrice(market)}
                    livePrices={livePrices}
                    onRemove={() => removeFavorite(market.id)}
                    onNavigate={() => onNavigateToMarket?.(market.id)}
                  />
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

interface MarketSearchItemProps {
  market: SearchableMarket;
  isFavorite: boolean;
  onToggle: () => void;
  price: number;
}

function MarketSearchItem({
  market,
  isFavorite,
  onToggle,
  price,
}: MarketSearchItemProps) {
  return (
    <div className="group px-4 py-3 hover:bg-slate-800/50 transition-colors">
      <div className="flex items-start justify-between gap-3">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <span className="font-semibold text-sm text-white">
              {market.symbol}
            </span>
            <span className="text-xs font-mono text-slate-400">
              ${price.toFixed(3)}
            </span>
          </div>
          <p className="text-xs text-slate-500 truncate" title={market.name}>
            {market.name}
          </p>
        </div>
        <button
          onClick={onToggle}
          className={clsx(
            "p-2 rounded-lg transition-all",
            isFavorite
              ? "bg-yellow-500/20 text-yellow-500 hover:bg-yellow-500/30"
              : "bg-slate-800 text-slate-400 hover:text-yellow-500 hover:bg-slate-700"
          )}
        >
          <Star size={16} className={isFavorite ? "fill-current" : ""} />
        </button>
      </div>
    </div>
  );
}

interface FavoriteMarketItemProps {
  market: FavoriteMarket;
  price: number;
  livePrices?: Record<string, number>;
  onRemove: () => void;
  onNavigate: () => void;
}

function FavoriteMarketItem({
  market,
  price,
  livePrices,
  onRemove,
  onNavigate,
}: FavoriteMarketItemProps) {
  const isLive =
    livePrices &&
    (livePrices[market.id] !== undefined ||
      (market.marketData?.outcomes?.length &&
        livePrices[market.marketData.outcomes[0].id] !== undefined));

  return (
    <div className="group px-4 py-3 hover:bg-slate-800/50 transition-colors">
      <div className="flex items-start justify-between gap-3">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <Star size={12} className="text-yellow-500 fill-yellow-500" />
            <span className="font-semibold text-sm text-white">
              {market.symbol}
            </span>
            <span
              className={clsx(
                "text-xs font-mono",
                isLive ? "text-emerald-400" : "text-slate-400"
              )}
            >
              ${price.toFixed(3)}
            </span>
            {isLive && (
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
            )}
          </div>
          <p className="text-xs text-slate-500 truncate" title={market.name}>
            {market.name}
          </p>
        </div>
        <div className="flex items-center gap-1">
          <button
            onClick={onNavigate}
            className="p-2 rounded-lg bg-slate-800 text-slate-400 hover:text-indigo-400 hover:bg-slate-700 transition-all"
            title="Open in Markets"
          >
            <ArrowUpRight size={14} />
          </button>
          <button
            onClick={onRemove}
            className="p-2 rounded-lg bg-slate-800 text-slate-400 hover:text-rose-400 hover:bg-slate-700 transition-all"
            title="Remove from favorites"
          >
            <X size={14} />
          </button>
        </div>
      </div>
    </div>
  );
}
