import { useState, useEffect } from "react";
import { PriceChart } from "./components/charts/PriceChart";
import { OrderBook } from "./components/dashboard/OrderBook";
import ScraperTab from "./components/ScraperTab";
import { useArena } from "./hooks/useArena";
import { Play, Square, RotateCcw, Activity, LineChart, Download } from "lucide-react";
import { listen } from "@tauri-apps/api/event";
import clsx from "clsx";

function App() {
  const { data: arenaData, history, isRunning, start, stop } = useArena();
  const [logs, setLogs] = useState<string[]>([]);
  const [activeTab, setActiveTab] = useState<'simulation' | 'scraper'>('simulation');

  // Listen for logs
  useEffect(() => {
    const unlisten = listen("arena-update", (event: any) => {
      if (event.payload.message) {
        setLogs(prev => [event.payload.message, ...prev].slice(0, 50));
      }
    });

    return () => {
      unlisten.then(f => f());
    };
  }, []);

  const handleStart = async () => {
    start();
  };

  const handleStop = async () => {
    stop();
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-50 font-sans selection:bg-indigo-500/30 flex flex-col h-screen">
      {/* Header */}
      <header className="flex items-center justify-between border-b border-slate-800 px-4 py-3 bg-slate-950 z-10">
        <div className="flex items-center gap-6">
          <div className="flex items-center gap-3">
            <Activity className="text-indigo-400 w-8 h-8" />
            <h1 className="text-xl font-bold tracking-tight">nglab <span className="text-indigo-400">Arena</span></h1>
          </div>

          <nav className="flex items-center gap-1 bg-slate-900 rounded-lg p-1 border border-slate-800">
            <button
              onClick={() => setActiveTab('simulation')}
              className={clsx(
                "flex items-center gap-2 px-3 py-1.5 rounded-md text-sm font-medium transition-all",
                activeTab === 'simulation'
                  ? "bg-indigo-600 text-white shadow-sm"
                  : "text-slate-400 hover:text-white hover:bg-slate-800"
              )}
            >
              <LineChart size={16} /> Simulation
            </button>
            <button
              onClick={() => setActiveTab('scraper')}
              className={clsx(
                "flex items-center gap-2 px-3 py-1.5 rounded-md text-sm font-medium transition-all",
                activeTab === 'scraper'
                  ? "bg-indigo-600 text-white shadow-sm"
                  : "text-slate-400 hover:text-white hover:bg-slate-800"
              )}
            >
              <Download size={16} /> Scraper
            </button>
          </nav>
        </div>

        {activeTab === 'simulation' && (
          <div className="flex gap-2">
            {!isRunning ? (
              <button
                onClick={handleStart}
                className="flex items-center gap-2 bg-indigo-600 hover:bg-indigo-500 px-3 py-1.5 rounded-lg transition-colors font-medium text-sm shadow-lg shadow-indigo-500/20"
              >
                <Play size={16} fill="currentColor" /> Start
              </button>
            ) : (
              <button
                onClick={handleStop}
                className="flex items-center gap-2 bg-rose-600 hover:bg-rose-500 px-3 py-1.5 rounded-lg transition-colors font-medium text-sm"
              >
                <Square size={16} fill="currentColor" /> Stop
              </button>
            )}
            <button className="p-1.5 bg-slate-800 hover:bg-slate-700 rounded-lg transition-colors border border-slate-700">
              <RotateCcw size={16} />
            </button>
          </div>
        )}
      </header>

      {/* Main Content */}
      <main className="flex-1 overflow-hidden p-4">
        {activeTab === 'simulation' ? (
          <div className="grid grid-cols-12 gap-6 h-full">
            {/* Left: Charts & Orderbook */}
            <div className="col-span-8 flex flex-col gap-6 h-full overflow-hidden">
              <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-4 flex-1 relative overflow-hidden flex flex-col">
                <h3 className="text-sm font-semibold text-slate-400 mb-2 uppercase tracking-wider">Live Price (Polymarket)</h3>
                <div className="flex-1 w-full bg-slate-950/50 rounded-lg overflow-hidden">
                  <PriceChart data={history} />
                </div>
              </div>

              <div className="h-64 bg-slate-900/50 border border-slate-800 rounded-xl p-4 overflow-hidden">
                <h3 className="text-sm font-semibold text-slate-400 mb-2 uppercase tracking-wider">Order Book Depth</h3>
                <OrderBook book={arenaData?.orderbook || { bids: {}, asks: {}, timestamp: 0 }} />
              </div>
            </div>

            {/* Right: Stats & Logs */}
            <div className="col-span-4 flex flex-col gap-6 h-full overflow-hidden">
              {/* Stats Panel */}
              <div className="bg-indigo-600/10 border border-indigo-500/20 rounded-xl p-6">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-xs text-indigo-300 uppercase font-bold">Current Step</p>
                    <p className="text-2xl font-mono">{arenaData?.step || 0}</p>
                  </div>
                  <div>
                    <p className="text-xs text-indigo-300 uppercase font-bold">Portfolio Value</p>
                    <p className={`text-2xl font-mono ${(arenaData?.portfolio_value || 0) >= 10000 ? 'text-emerald-400' : 'text-rose-400'}`}>
                      {arenaData?.portfolio_value?.toFixed(2) || "10000.00"}
                    </p>
                  </div>
                </div>
              </div>

              {/* Log Panel */}
              <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-4 flex-1 flex flex-col overflow-hidden">
                <h3 className="text-sm font-semibold text-slate-400 mb-3 uppercase tracking-wider">Agent Logs</h3>
                <div className="flex-1 overflow-y-auto font-mono text-xs space-y-1 pr-2 scrollbar-thin scrollbar-thumb-slate-700">
                  {logs.length === 0 && <p className="text-slate-600 italic">Waiting for simulation data...</p>}
                  {logs.map((log, i) => (
                    <div key={i} className="text-slate-300 border-l border-slate-700 pl-2 py-0.5">
                      <span className="text-indigo-400 opacity-50 mr-2">[{new Date().toLocaleTimeString()}]</span>
                      {log}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        ) : (
          <div className="h-full bg-slate-900/50 border border-slate-800 rounded-xl overflow-hidden">
            <ScraperTab />
          </div>
        )}
      </main>
    </div>
  );
}

export default App;