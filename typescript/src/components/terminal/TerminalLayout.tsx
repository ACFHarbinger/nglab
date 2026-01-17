/**
 * @module components/terminal/TerminalLayout
 * @description High-fidelity trading terminal layout with chart, order book, trades, and execution form.
 */
import { useState, useEffect } from "react";
/**
 * @module components/terminal/MarketSidebar
 * @description Searchable sidebar for quickly switching between different trading pairs/markets.
 */
import { MarketSidebar } from "./MarketSidebar";
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
import { usePolymarket } from "../../hooks/usePolymarket";
import { Wallet, Settings } from "lucide-react";

// Mock Data Generators
const generateMockMarkets = () => [
  {
    id: "1",
    symbol: "TRUMP",
    name: "Trump 2024 Election Win",
    price: 0.52,
    change24h: 2.5,
    volume24h: 154000,
    isFavorite: true,
  },
  {
    id: "2",
    symbol: "Biden",
    name: "Biden 2024 Election Win",
    price: 0.12,
    change24h: -5.1,
    volume24h: 45000,
  },
  {
    id: "3",
    symbol: "ETH",
    name: "ETH > $3k by March",
    price: 0.78,
    change24h: 1.2,
    volume24h: 89000,
  },
  {
    id: "4",
    symbol: "FED",
    name: "Fed Rate Cut in May",
    price: 0.45,
    change24h: 0.0,
    volume24h: 22000,
  },
  {
    id: "5",
    symbol: "BTC",
    name: "BTC > $100k in 2024",
    price: 0.33,
    change24h: -1.5,
    volume24h: 67000,
    isFavorite: true,
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

export function TerminalLayout() {
  const { livePrices } = usePolymarket();
  const [selectedMarketId, setSelectedMarketId] = useState("1");
  const [chartData, setChartData] = useState<any[]>([]);
  const [orderBook, setOrderBook] = useState<{ bids: any; asks: any }>({
    bids: {},
    asks: {},
  });
  const [recentTrades, setRecentTrades] = useState<Trade[]>([]);

  const markets = generateMockMarkets();
  const selectedMarket =
    markets.find((m) => m.id === selectedMarketId) || markets[0];

  // Live price update or fallback to mock
  const currentPrice = livePrices[selectedMarketId] || selectedMarket.price;

  useEffect(() => {
    // Init mock data on market switch
    setChartData(generateMockHistory(selectedMarket.price));
    setOrderBook(generateMockOrderBook(selectedMarket.price));
    setRecentTrades(generateMockTrades(selectedMarket.price));
  }, [selectedMarketId]);

  useEffect(() => {
    // Update chart with live price if streaming
    if (livePrices[selectedMarketId]) {
      setChartData((prev) => [
        ...prev,
        {
          time: Math.floor(Date.now() / 1000),
          value: livePrices[selectedMarketId],
        },
      ]);
    }
  }, [livePrices, selectedMarketId]);

  return (
    <div className="flex h-full w-full bg-slate-950 text-slate-200 overflow-hidden">
      {/* 1. Market Sidebar (List) */}
      <MarketSidebar
        markets={markets}
        activeMarketId={selectedMarketId}
        onSelectMarket={setSelectedMarketId}
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
                ${currentPrice.toFixed(3)}
              </span>
              <span
                className={`text-xs ${selectedMarket.change24h >= 0 ? "text-green-400" : "text-rose-400"}`}
              >
                {selectedMarket.change24h > 0 ? "+" : ""}
                {selectedMarket.change24h}%
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
        <div className="h-1/2">
          <RecentTradesWidget trades={recentTrades} />
        </div>
      </div>

      {/* 4. Far Right: Trading Form (Buy/Sell) */}
      <div className="w-80">
        <TradingFormWidget
          symbol={selectedMarket.symbol}
          currentPrice={currentPrice}
        />
      </div>
    </div>
  );
}
