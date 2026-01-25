/**
 * @module components/terminal/TerminalLayout
 * @description High-fidelity trading terminal layout with chart, order book, trades, and execution form.
 */
import { useState, useEffect, useMemo } from "react";
/**
 * @module components/terminal/MarketSidebar
 * @description Searchable sidebar for quickly switching between different trading pairs/markets.
 */
import { MarketSidebar, Market } from "./MarketSidebar";
/**
 * @module components/terminal/TerminalChart
 * @description Real-time price chart visualization for a selected market.
 */
import { TerminalChart } from "./TerminalChart";
/**
 * @module components/terminal/OrderBookWidget
 * @description Vertical limit order book visualization optimized for the trading terminal view.
 */
import { OrderBookWidget } from "./OrderBookWidget";
/**
 * @module components/terminal/RecentTradesWidget
 * @description Displays a list of the most recent trades for the active market.
 */
import { RecentTradesWidget, Trade } from "./RecentTradesWidget";
/**
 * @module components/terminal/TradingFormWidget
 * @description Form for placing buy/sell orders for the selected market.
 */
import { TradingFormWidget } from "./TradingFormWidget";
import { AlgoOrderWidget } from "./AlgoOrderWidget";
import { MarketMakerWidget } from "./MarketMakerWidget";
import { useArena } from "../../hooks/useArena";
import { MarketMetadata } from "../../hooks/usePolymarket";
import { FavoriteMarket } from "../../hooks/useFavorites";
import { Wallet, Settings } from "lucide-react";
import { invoke } from "@tauri-apps/api/core";

// Mock Data Generators
const generateMockMarkets = () => [
  {
    id: "1",
    symbol: "WARSH",
    name: "Kevin Warsh for Fed Chair",
    price: 0.65,
    change24h: 4.2,
    volume24h: 208000000,
    // Real Token IDs for testing (2026)
    marketData: {
      id: "will-trump-nominate-kevin-warsh-as-the-next-fed-chair",
      outcomes: [
        {
          id: "51338236787729560681434534660841415073585974762690814047670810862722808070955",
          name: "Yes",
        },
        {
          id: "18289842382539867639079362738467334752951741961393928566628307174343542320349",
          name: "No",
        },
      ],
    },
  },
  {
    id: "2",
    symbol: "HASSETT",
    name: "Kevin Hassett for Fed Chair",
    price: 0.22,
    change24h: -1.1,
    volume24h: 85000000,
    marketData: {
      id: "will-trump-nominate-kevin-hassett-as-the-next-fed-chair",
      outcomes: [
        {
          id: "34551606549875928972193520396544368029176529083448203019529657908155427866742",
          name: "Yes",
        },
        {
          id: "22802130763821766047382314926654345322953005062130920361011056163048911488589",
          name: "No",
        },
      ],
    },
  },
  {
    id: "3",
    symbol: "ETH",
    name: "ETH > $10k by Dec 2026",
    price: 0.48,
    change24h: 3.2,
    volume24h: 489000000,
    marketData: {
      id: "will-ethereum-reach-10000-by-december-31-2026",
      outcomes: [
        {
          id: "100865557115198861945068078147655261264339702053336500548943844953071484972262",
          name: "Yes",
        },
        {
          id: "110234250181945907038686474667990126495798306823136531546168897766302386074534",
          name: "No",
        },
      ],
    },
  },
  {
    id: "4",
    symbol: "FED",
    name: "Fed Rate Cut in Jan 2026",
    price: 0.85,
    change24h: 0.2,
    volume24h: 398000000,
    marketData: {
      id: "fed-decreases-interest-rates-by-25-bps-after-january-2026-meeting",
      outcomes: [
        {
          id: "92703761682322480664976766247614127878023988651992837287050266308961660624165",
          name: "Yes",
        },
        {
          id: "48193521645113703700467246669338225849301704920590102230072263970163239985027",
          name: "No",
        },
      ],
    },
  },
  {
    id: "5",
    symbol: "BTC",
    name: "BTC > $100k in 2026",
    price: 0.53,
    change24h: -0.5,
    volume24h: 967000000,
    marketData: {
      id: "will-bitcoin-reach-100000-by-december-31-2026-571",
      outcomes: [
        {
          id: "56078938060096976448086754249497300447360333783952000147427828224794011030104",
          name: "Yes",
        },
        {
          id: "11291662904897713174667903388388696640643610556195928998276904135282270136756",
          name: "No",
        },
      ],
    },
  },
];

const generateMockHistory = (currentPrice: number) => {
  const data = [];
  let price = currentPrice;
  const now = Math.floor(Date.now() / 1000);
  for (let i = 1000; i >= 0; i--) {
    data.push({ time: now - i * 60, value: price });
    price = price + (Math.random() - 0.5) * 0.01;
  }
  return data;
};

const generateMockOrderBook = (price: number) => {
  const bids: any = {};
  const asks: any = {};
  for (let i = 1; i <= 15; i++) {
    const bidPrice = price - i * 0.005;
    const askPrice = price + i * 0.005;
    bids[i] = { price: bidPrice, total_quantity: Math.random() * 1000 + 100 };
    asks[i] = { price: askPrice, total_quantity: Math.random() * 1000 + 100 };
  }
  return { bids, asks };
};

const generateMockTrades = (price: number): Trade[] => {
  const trades: Trade[] = [];
  const now = Math.floor(Date.now() / 1000);
  for (let i = 0; i < 20; i++) {
    trades.push({
      id: `trade-${i}`,
      price: price + (Math.random() - 0.5) * 0.01,
      amount: Math.random() * 500,
      size: Math.random() * 1000,
      side: Math.random() > 0.5 ? "buy" : "sell",
      timestamp: now - i * 30, // Every 30s
    });
  }
  return trades;
};

/**
 * Props for the TerminalLayout component.
 */
interface Props {
  /** Real-time live price map. */
  livePrices: Record<string, number>;
  /** Metadata for the currently active market. */
  activeMarket: MarketMetadata | null;
  /** Function to start streaming market data. */
  startStream?: (
    marketSource: string,
    metadata: MarketMetadata,
  ) => Promise<void>;
  /** Function to stop the current stream. */
  stopStream?: () => Promise<void>;
  /** Set of favorite market IDs. */
  favoriteIds: Set<string>;
  /** Array of favorite market data. */
  favorites: FavoriteMarket[];
  /** Callback to toggle favorite status. */
  toggleFavorite: (market: any) => void;
}

/**
 * The core trading terminal interface.
 * Integrates Chart, Order Book, Recent Trades, and Trading execution form.
 * Manages local state for mock data and coordinates live data updates.
 */
export function TerminalLayout({
  livePrices,
  activeMarket,
  startStream,
  stopStream,
  favoriteIds,
  favorites,
  toggleFavorite,
}: Props) {
  const { data: arenaData } = useArena();
  const [chartData, setChartData] = useState<any[]>([]);
  const [orderBook, setOrderBook] = useState<{ bids: any; asks: any }>({
    bids: {},
    asks: {},
  });
  const [recentTrades, setRecentTrades] = useState<Trade[]>([]);
  const [trendingMarkets, setTrendingMarkets] = useState<any[]>([]);

  // Always use fresh mock data + trending state + favorites
  const markets = useMemo(() => {
    const mockMarkets = generateMockMarkets();
    const mappedFavorites: Market[] = (favorites || []).map((f) => ({
      id: f.id,
      symbol: f.symbol,
      name: f.name,
      price: 0.5, // Default for favorites not currently streaming
      change24h: 0,
      volume24h: 0,
      isFavorite: true,
      marketData: f.marketData,
    }));

    // Combine all sources
    const combined = [...mockMarkets, ...trendingMarkets, ...mappedFavorites];

    // Deduplicate by ID, prioritizing later entries (mappedFavorites)
    const uniqueMarketsMap = new Map<string, Market>();
    combined.forEach((m) => {
      uniqueMarketsMap.set(m.id, {
        ...m,
        isFavorite: (favoriteIds || new Set()).has(m.id),
      });
    });

    return Array.from(uniqueMarketsMap.values());
  }, [trendingMarkets, favoriteIds, favorites]);

  // Handle initialization of selection from external activeMarket
  const [selectedMarketId, setSelectedMarketId] = useState(() => {
    if (activeMarket?.title) {
      const match = markets.find((m) => m.name === activeMarket.title);
      return match ? match.id : "1";
    }
    return "1";
  });

  // External sync: If activeMarket changes from OUTSIDE (e.g. Navigation), sync local state.
  useEffect(() => {
    if (activeMarket?.title) {
      const match = markets.find((m) => m.name === activeMarket.title);
      if (match && match.id !== selectedMarketId) {
        console.log("🔄 External sync triggered in terminal:", match.id);
        setSelectedMarketId(match.id);
      }
    }
  }, [activeMarket?.title, markets]); // Dependency on title ensures it only runs on actual changes

  useEffect(() => {
    fetchTrending();
  }, []);

  const fetchTrending = async () => {
    try {
      const response: any = await invoke("get_trending_markets", { limit: 10 });
      if (response.success && response.data) {
        // Map backend events to frontend market structure
        const trending = (response.data || []).map(
          (event: any, index: number) => {
            // Flatten: Use first market of the event or specific logic
            const market = event.markets?.[0]; // Simplification
            return {
              id: market?.id || event.id,
              symbol: event.title.substring(0, 10).toUpperCase(), // Naive symbol
              name: event.title,
              price: 0.5, // Default or fetch
              change24h: 0,
              volume24h: 10000 - index * 100, // Fake volume desc
              isFavorite: false,
              // Store full data for streaming
              marketData: market,
            };
          },
        );

        // Merge with mocks or replace
        if (trending.length > 0) {
          setTrendingMarkets(trending);
        }
      }
    } catch (e) {
      console.error("Failed to fetch trending markets", e);
    }
  };

  const selectedMarket: any = markets.find((m) => m.id === selectedMarketId) ||
    markets[0] || { symbol: "...", price: 0, change24h: 0 };

  // Start stream when market is selected
  useEffect(() => {
    if (startStream && selectedMarket && selectedMarket.id) {
      // Use the actual market ID from marketData if it exists (for both mocks and trending)
      const marketSource = selectedMarket.marketData?.id || selectedMarket.id;

      console.log(
        `📡 Switching terminal stream to: ${marketSource} (${selectedMarket.name})`,
      );

      // We explicitly stop the previous stream before starting a new one to ensure clean state
      const runStream = async () => {
        if (stopStream) await stopStream();
        await startStream(marketSource, selectedMarket.marketData);
      };

      runStream().catch((e) =>
        console.error("Terminal stream transition failed", e),
      );
    }

    return () => {
      if (stopStream) {
        stopStream().catch((e) =>
          console.error("Cleanup stream stop failed", e),
        );
      }
    };
  }, [selectedMarketId, selectedMarket?.name, startStream, stopStream]);

  // Live price update logic
  let currentPrice = selectedMarket?.price || 0;

  if (livePrices && selectedMarket) {
    // 1. Check direct mapping (simple)
    if (livePrices[selectedMarketId]) {
      currentPrice = livePrices[selectedMarketId];
    }
    // 2. Check by Token/Outcome ID (Polymarket)
    else if (selectedMarket?.marketData?.outcomes?.length > 0) {
      // Assume first outcome is the main one (Yes)
      const yesId = selectedMarket.marketData.outcomes[0].id;
      if (livePrices[yesId]) {
        currentPrice = livePrices[yesId];
      }
    }
  }

  // Debug Overlay logic
  const lastAssetId = Object.keys(livePrices || {}).pop() || "None";
  const lastPrice = lastAssetId !== "None" ? livePrices[lastAssetId] : null;

  const DebugOverlay = () => {
    return (
      <div className="fixed bottom-24 right-12 bg-slate-900 border-2 border-emerald-500 text-[11px] text-emerald-400 p-4 rounded-xl z-[9999] pointer-events-none shadow-[0_0_30px_rgba(16,185,129,0.3)] backdrop-blur-md min-w-[220px]">
        <div className="font-bold uppercase tracking-widest border-b border-emerald-500/20 pb-2 mb-3 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div
              className={`w-2 h-2 rounded-full ${Object.keys(livePrices || {}).length > 0 ? "bg-emerald-500 animate-pulse" : "bg-red-500"}`}
            />
            <span>Live Stream</span>
          </div>
          <span className="text-[9px] opacity-50 px-2 py-0.5 rounded-full bg-emerald-500/10 border border-emerald-500/20">
            WS-CLOB-2026
          </span>
        </div>
        <div className="space-y-2">
          <div className="flex justify-between items-center">
            <span className="opacity-60">Active Assets:</span>
            <span className="text-white font-bold">
              {Object.keys(livePrices || {}).length}
            </span>
          </div>
          <div className="flex justify-between items-center border-t border-emerald-500/10 pt-2 mt-2">
            <span className="opacity-60 text-emerald-400">Last Event:</span>
            <span className="text-white truncate max-w-[100px]">
              {lastAssetId.substring(0, 8)}...
            </span>
          </div>
          {isFinite(lastPrice as number) && lastPrice !== null && (
            <div className="flex justify-between items-center text-[9px] opacity-40">
              <span>Last Price:</span>
              <span>${(lastPrice as number).toFixed(4)}</span>
            </div>
          )}
        </div>
      </div>
    );
  };

  useEffect(() => {
    // Init mock data on market switch
    setChartData(generateMockHistory(selectedMarket.price));
    setOrderBook(generateMockOrderBook(selectedMarket.price));
    setRecentTrades(generateMockTrades(selectedMarket.price));
  }, [selectedMarketId, selectedMarket.price]);

  useEffect(() => {
    // Update chart with live price
    // We use the calculated currentPrice for the chart update
    if (
      Object.keys(livePrices || {}).length > 0 &&
      isFinite(currentPrice) &&
      currentPrice !== selectedMarket.price
    ) {
      setChartData((prev: any[]) => {
        const now = Math.floor(Date.now() / 1000);
        const lastPoint = prev[prev.length - 1];

        // Lightweight-charts requires unique, ascending timestamps.
        // If we get multiple updates in the same second, we update the last point's value instead of pushing a new one.
        if (lastPoint && lastPoint.time === now) {
          const updated = [...prev];
          updated[updated.length - 1] = { ...lastPoint, value: currentPrice };
          return updated;
        }

        return [
          ...prev,
          {
            time: now,
            value: currentPrice,
          },
        ];
      });
    }
  }, [livePrices, currentPrice]);

  return (
    <div className="flex h-full w-full bg-slate-950 text-slate-200 overflow-hidden relative">
      <DebugOverlay />
      {/* 1. Market Sidebar (List) */}
      <MarketSidebar
        markets={markets}
        activeMarketId={selectedMarketId}
        onSelectMarket={setSelectedMarketId}
        livePrices={livePrices}
        favoriteIds={favoriteIds}
        onToggleFavorite={toggleFavorite}
      />

      {/* 2. Main Center Area (Chart & Header) */}
      <div className="flex-1 flex flex-col min-w-0 border-r border-slate-800">
        {/* Header */}
        <header className="flex justify-between items-center px-4 py-3 border-b border-slate-800 bg-slate-950">
          <div className="flex items-center gap-4">
            <h1 className="text-xl font-bold flex items-center gap-2">
              {selectedMarket.symbol}{" "}
              <span className="text-slate-500 font-normal text-sm">/ USD</span>
            </h1>
            <div className="flex flex-col">
              <span className="text-2xl font-mono font-bold text-white">
                ${isFinite(currentPrice) ? currentPrice.toFixed(3) : "0.000"}
              </span>
              <span
                className={`text-xs ${selectedMarket.change24h >= 0 ? "text-green-400" : "text-rose-400"}`}
              >
                {isFinite(selectedMarket.change24h)
                  ? selectedMarket.change24h > 0
                    ? "+"
                    : ""
                  : ""}
                {isFinite(selectedMarket.change24h)
                  ? selectedMarket.change24h
                  : "0"}
                %
              </span>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <button className="flex items-center gap-2 px-3 py-1.5 bg-indigo-600 hover:bg-indigo-500 rounded text-sm font-medium transition-colors">
              <Wallet size={16} /> Connect
            </button>
            <button className="p-2 hover:bg-slate-800 rounded text-slate-400">
              <Settings size={18} />
            </button>
          </div>
        </header>

        {/* Chart Area */}
        <div className="flex-1 relative bg-slate-950">
          <TerminalChart
            data={chartData}
            color={selectedMarket.change24h >= 0 ? "#22c55e" : "#f43f5e"}
          />
        </div>
      </div>

      {/* 3. Right Column: Order Book & Trades (Stacked) */}
      <div className="flex flex-col w-72 border-r border-slate-800">
        {/* Order Book (Top 50%) */}
        <div className="h-1/2 border-b border-slate-800">
          <OrderBookWidget bids={orderBook.bids} asks={orderBook.asks} />
        </div>
        {/* Recent Trades (Bottom 50%) */}
        <div className="flex-1">
          <RecentTradesWidget trades={recentTrades} />
        </div>
        {/* Algo Execution */}
        <div className="h-48 border-t border-slate-800">
          <AlgoOrderWidget
            algo_orders={arenaData?.algo_orders || []}
            current_step={arenaData?.step || 0}
          />
        </div>
        {/* Market Maker Dashboard */}
        <div className="h-48 border-t border-slate-800">
          <MarketMakerWidget
            mm_active={arenaData?.mm_active || false}
            mm_realized_pnl={arenaData?.mm_realized_pnl || 0}
            inventory={arenaData?.position || 0}
          />
        </div>
      </div>

      {/* 4. Far Right: Trading Form (Buy/Sell) */}
      <div className="w-80">
        <TradingFormWidget
          symbol={selectedMarket.symbol}
          currentPrice={currentPrice}
          outcomes={selectedMarket.marketData?.outcomes}
          livePrices={livePrices}
        />
      </div>
    </div>
  );
}
