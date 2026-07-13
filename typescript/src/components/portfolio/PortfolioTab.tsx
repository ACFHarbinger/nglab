import { useState } from "react";
import { PortfolioAllocation } from "./PortfolioAllocation";
import { 
  TrendingUp, 
  TrendingDown, 
  ArrowUpRight, 
  ArrowDownRight, 
  List, 
  History, 
  LayoutGrid,
  Download
} from "lucide-react";
import clsx from "clsx";

/**
 * PortfolioTab Component
 * 
 * Provides a specialized view for managing real-time positions, 
 * historical trade logs, and portfolio optimization.
 */
export default function PortfolioTab() {
  const [view, setView] = useState<"allocation" | "positions" | "history">("positions");

  // Mock positions data (Quick Win #4)
  const [positions] = useState([
    { 
      id: "BTC-1", 
      symbol: "BTC", 
      side: "LONG", 
      size: 0.5, 
      entry: 64200.50, 
      current: 65120.20, 
      pnl: 459.85, 
      pnlPct: 1.43,
      leverage: "1x",
      margin: 32100.25
    },
    { 
      id: "ETH-1", 
      symbol: "ETH", 
      side: "LONG", 
      size: 10.0, 
      entry: 3450.20, 
      current: 3380.15, 
      pnl: -700.50, 
      pnlPct: -2.03,
      leverage: "1x",
      margin: 34502.00
    },
    { 
      id: "SOL-1", 
      symbol: "SOL", 
      side: "SHORT", 
      size: 150.0, 
      entry: 145.20, 
      current: 142.10, 
      pnl: 465.00, 
      pnlPct: 2.13,
      leverage: "2x",
      margin: 10890.00
    }
  ]);

  /**
   * Quick Win #9: Export to CSV
   * Generates a CSV from the current positions and triggers a download.
   */
  const exportToCSV = () => {
    const headers = ["Asset", "Side", "Size", "Entry Price", "Current Price", "Unrealized P&L", "P&L %"];
    const rows = positions.map(p => [
      p.symbol, p.side, p.size, p.entry, p.current, p.pnl, p.pnlPct
    ]);
    
    const csvContent = [headers, ...rows]
      .map(e => e.join(","))
      .join("\n");
      
    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const link = document.createElement("a");
    const url = URL.createObjectURL(blob);
    link.setAttribute("href", url);
    link.setAttribute("download", `nglab_positions_${new Date().toISOString().split('T')[0]}.csv`);
    link.style.visibility = "hidden";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="flex flex-col h-full bg-slate-950 overflow-hidden">
      {/* Tab Header / Navigation */}
      <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800 bg-slate-900/40">
        <div className="flex items-center gap-1 bg-slate-900 p-1 rounded-lg border border-slate-800">
          <TabButton 
            active={view === "positions"} 
            onClick={() => setView("positions")}
            icon={List}
            label="Positions"
          />
          <TabButton 
            active={view === "allocation"} 
            onClick={() => setView("allocation")}
            icon={LayoutGrid}
            label="Allocation & Risk"
          />
          <TabButton 
            active={view === "history"} 
            onClick={() => setView("history")}
            icon={History}
            label="Trade History"
          />
        </div>

        <div className="flex items-center gap-3">
          <button 
            onClick={exportToCSV}
            className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-medium transition-colors border border-slate-700"
          >
            <Download size={14} />
            Export CSV
          </button>
          <div className="h-6 w-px bg-slate-800 mx-1" />
          <div className="flex items-center gap-2">
            <span className="text-xs text-slate-500 uppercase font-bold">Unrealized P&L</span>
            <span className={clsx("text-sm font-mono font-bold", (224.35 >= 0) ? "text-emerald-400" : "text-rose-400")}>
              +$224.35 (+0.28%)
            </span>
          </div>
        </div>
      </div>

      {/* Content Area */}
      <div className="flex-1 overflow-hidden">
        {view === "positions" && (
          <div className="h-full p-6 overflow-y-auto">
            <div className="bg-slate-900 rounded-xl border border-slate-800 overflow-hidden">
              <table className="w-full text-left border-collapse">
                <thead>
                  <tr className="bg-slate-950/50 border-b border-slate-800">
                    <th className="px-4 py-3 text-[10px] uppercase font-bold text-slate-500">Asset</th>
                    <th className="px-4 py-3 text-[10px] uppercase font-bold text-slate-500 text-right">Side</th>
                    <th className="px-4 py-3 text-[10px] uppercase font-bold text-slate-500 text-right">Size</th>
                    <th className="px-4 py-3 text-[10px] uppercase font-bold text-slate-500 text-right">Entry Price</th>
                    <th className="px-4 py-3 text-[10px] uppercase font-bold text-slate-500 text-right">Current Price</th>
                    <th className="px-4 py-3 text-[10px] uppercase font-bold text-slate-500 text-right">Unrealized P&L</th>
                    <th className="px-4 py-3 text-[10px] uppercase font-bold text-slate-500 text-right">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/50">
                  {positions.map((pos) => (
                    <tr key={pos.id} className="hover:bg-slate-800/30 transition-colors group">
                      <td className="px-4 py-4">
                        <div className="flex items-center gap-3">
                          <div className="w-8 h-8 rounded bg-slate-800 flex items-center justify-center font-bold text-xs text-slate-400">
                            {pos.symbol[0]}
                          </div>
                          <div>
                            <div className="font-bold text-slate-200">{pos.symbol}-USD</div>
                            <div className="text-[10px] text-slate-500 font-mono">ID: {pos.id}</div>
                          </div>
                        </div>
                      </td>
                      <td className="px-4 py-4 text-right">
                        <span className={clsx(
                          "text-[10px] font-bold px-2 py-0.5 rounded",
                          pos.side === "LONG" ? "bg-emerald-500/10 text-emerald-400" : "bg-rose-500/10 text-rose-400"
                        )}>
                          {pos.side}
                        </span>
                      </td>
                      <td className="px-4 py-4 text-right font-mono text-sm text-slate-300">
                        {pos.size.toFixed(2)}
                      </td>
                      <td className="px-4 py-4 text-right font-mono text-sm text-slate-400">
                        ${pos.entry.toLocaleString()}
                      </td>
                      <td className="px-4 py-4 text-right font-mono text-sm text-slate-200">
                        ${pos.current.toLocaleString()}
                      </td>
                      <td className="px-4 py-4 text-right">
                        <div className={clsx("font-mono font-bold text-sm", pos.pnl >= 0 ? "text-emerald-400" : "text-rose-400")}>
                          {pos.pnl >= 0 ? "+" : ""}{pos.pnl.toLocaleString(undefined, { minimumFractionDigits: 2 })}
                        </div>
                        <div className={clsx("text-[10px] font-mono", pos.pnlPct >= 0 ? "text-emerald-500/70" : "text-rose-500/70")}>
                          {pos.pnlPct >= 0 ? "+" : ""}{pos.pnlPct}%
                        </div>
                      </td>
                      <td className="px-4 py-4 text-right">
                        <button className="text-[10px] font-bold text-indigo-400 hover:text-indigo-300 opacity-0 group-hover:opacity-100 transition-opacity">
                          CLOSE POSITION
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {view === "allocation" && (
          <div className="h-full">
            <PortfolioAllocation />
          </div>
        )}

        {view === "history" && (
          <div className="h-full flex items-center justify-center text-slate-500 italic">
            Trade history log coming soon...
          </div>
        )}
      </div>
    </div>
  );
}

function TabButton({ active, onClick, icon: Icon, label }: any) {
  return (
    <button
      onClick={onClick}
      className={clsx(
        "flex items-center gap-2 px-3 py-1.5 rounded-md text-xs font-bold transition-all",
        active 
          ? "bg-slate-800 text-white shadow-sm" 
          : "text-slate-500 hover:text-slate-300"
      )}
    >
      <Icon size={14} />
      {label}
    </button>
  );
}
