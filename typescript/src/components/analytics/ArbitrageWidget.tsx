import { useState, useEffect, useCallback, useMemo } from "react";
import { invoke } from "@tauri-apps/api/core";
import { 
  ArrowRightLeft, 
  RefreshCw, 
  AlertCircle,
  ChevronRight
} from "lucide-react";
import clsx from "clsx";

interface MarketData {
  symbol: string;
  last_price: number;
  best_bid: number;
  best_ask: number;
}

interface ArbitrageWidgetProps {
  defaultSymbol?: string;
}

export function ArbitrageWidget({ defaultSymbol = "BTC" }: ArbitrageWidgetProps) {
  const [symbol, setSymbol] = useState(defaultSymbol);
  const [prices, setPrices] = useState<Record<string, MarketData>>({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

  const fetchPrices = useCallback(async (targetSymbol: string) => {
    setLoading(true);
    setError(null);
    try {
      const results: Record<string, MarketData> = await invoke("get_all_exchange_prices", { 
        symbol: targetSymbol 
      });
      setPrices(results);
      setLastUpdated(new Date());
    } catch (err) {
      console.error("Failed to fetch arbitrage prices:", err);
      setError("Failed to fetch cross-exchange data");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchPrices(symbol);
    const interval = setInterval(() => fetchPrices(symbol), 30000); // Polling every 30s
    return () => clearInterval(interval);
  }, [symbol, fetchPrices]);

  // Arbitrage calculation
  const arbOpportunities = useMemo(() => {
    const exchangeNames = Object.keys(prices);
    if (exchangeNames.length < 2) return [];

    const opps: any[] = [];
    
    // Find absolute best bid and absolute best ask across all exchanges
    let bestBid = -Infinity;
    let bBidEx = "";
    let bestAsk = Infinity;
    let bAskEx = "";

    exchangeNames.forEach(ex => {
      const data = prices[ex];
      if (data.best_bid > bestBid) {
        bestBid = data.best_bid;
        bBidEx = ex;
      }
      if (data.best_ask < bestAsk) {
        bestAsk = data.best_ask;
        bAskEx = ex;
      }
    });

    // An opportunity exists if Best Bid > Best Ask (Spread is negative)
    if (bestBid > bestAsk) {
      const spread = bestBid - bestAsk;
      const spreadPct = (spread / bestAsk) * 100;
      opps.push({
        buyAt: bAskEx,
        sellAt: bBidEx,
        buyPrice: bestAsk,
        sellPrice: bestBid,
        profit: spread,
        profitPct: spreadPct
      });
    }

    return opps;
  }, [prices]);

  return (
    <div className="bg-slate-800 rounded-lg border border-slate-700 flex flex-col h-full overflow-hidden">
      <div className="p-4 border-b border-slate-700 flex justify-between items-center bg-slate-800/50">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-indigo-500/10 rounded-lg text-indigo-400">
            <ArrowRightLeft size={18} />
          </div>
          <div>
            <h3 className="text-sm font-bold text-slate-100 uppercase tracking-wider">Cross-Exchange Arb</h3>
            <p className="text-[10px] text-slate-500 uppercase font-medium">Real-time price discovery</p>
          </div>
        </div>
        
        <div className="flex items-center gap-2">
          <div className="bg-slate-950 border border-slate-800 rounded-md px-2 py-1 flex items-center gap-2">
             <input 
               type="text" 
               value={symbol}
               onChange={(e) => setSymbol(e.target.value.toUpperCase())}
               onKeyDown={(e) => e.key === 'Enter' && fetchPrices(symbol)}
               className="bg-transparent border-none text-xs text-white w-16 focus:outline-none uppercase font-mono"
             />
             <button 
               onClick={() => fetchPrices(symbol)}
               className={clsx("text-slate-500 hover:text-white transition-colors", loading && "animate-spin")}
             >
               <RefreshCw size={12} />
             </button>
          </div>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {error && (
          <div className="bg-rose-500/10 border border-rose-500/20 rounded-md p-3 flex items-center gap-2 text-xs text-rose-400">
            <AlertCircle size={14} />
            {error}
          </div>
        )}

        {/* Arb Alert */}
        {arbOpportunities.length > 0 ? (
          <div className="bg-emerald-500/10 border border-emerald-500/20 rounded-md p-4 animate-pulse">
            <div className="flex justify-between items-center mb-2">
              <span className="text-[10px] font-bold text-emerald-400 uppercase tracking-widest">Opportunity Detected</span>
              <span className="text-xs font-mono font-bold text-emerald-400">+{arbOpportunities[0].profitPct.toFixed(3)}%</span>
            </div>
            <div className="flex items-center justify-between gap-2">
              <div className="flex flex-col">
                <span className="text-[10px] text-slate-500 uppercase">Buy at</span>
                <span className="text-sm font-bold text-white tracking-wide">{arbOpportunities[0].buyAt}</span>
                <span className="text-xs font-mono text-slate-400">${arbOpportunities[0].buyPrice.toLocaleString()}</span>
              </div>
              <ChevronRight className="text-slate-700" size={16} />
              <div className="flex flex-col items-end">
                <span className="text-[10px] text-slate-500 uppercase">Sell at</span>
                <span className="text-sm font-bold text-white tracking-wide">{arbOpportunities[0].sellAt}</span>
                <span className="text-xs font-mono text-slate-400">${arbOpportunities[0].sellPrice.toLocaleString()}</span>
              </div>
            </div>
          </div>
        ) : !loading && (
          <div className="bg-slate-950/30 border border-slate-800/50 rounded-md p-3 text-center">
            <p className="text-[10px] text-slate-600 uppercase tracking-tight">No significant arbitrage found</p>
          </div>
        )}

        {/* Price Table */}
        <div className="rounded-md border border-slate-700/50 overflow-hidden">
          <table className="w-full text-xs">
            <thead className="bg-slate-950/50 text-slate-500 uppercase text-[9px] font-bold">
              <tr>
                <th className="px-3 py-2 text-left">Exchange</th>
                <th className="px-3 py-2 text-right">Last Price</th>
                <th className="px-3 py-2 text-right">Spread %</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800">
              {Object.entries(prices).map(([ex, data]) => {
                const spread = data.best_ask - data.best_bid;
                const spreadPct = (spread / data.best_bid) * 100;
                
                return (
                  <tr key={ex} className="hover:bg-slate-750 transition-colors">
                    <td className="px-3 py-2.5 font-medium text-slate-300">{ex}</td>
                    <td className="px-3 py-2.5 text-right font-mono text-white">
                      ${data.last_price.toLocaleString(undefined, { minimumFractionDigits: 2 })}
                    </td>
                    <td className={clsx(
                      "px-3 py-2.5 text-right font-mono",
                      spreadPct < 0.05 ? "text-emerald-400" : "text-slate-500"
                    )}>
                      {spreadPct.toFixed(3)}%
                    </td>
                  </tr>
                );
              })}
              {Object.keys(prices).length === 0 && !loading && (
                <tr>
                  <td colSpan={3} className="px-3 py-8 text-center text-slate-600 uppercase text-[10px]">
                    Enter a symbol to compare prices
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      <div className="p-2 bg-slate-950/50 border-t border-slate-800 flex justify-between items-center px-4">
        <span className="text-[9px] text-slate-600 uppercase font-bold">Status: {loading ? 'Syncing...' : 'Connected'}</span>
        {lastUpdated && (
          <span className="text-[9px] text-slate-600 font-mono">
            Updated: {lastUpdated.toLocaleTimeString()}
          </span>
        )}
      </div>
    </div>
  );
}

