import { useState, useEffect } from "react";
import { PriceChart } from "./components/charts/PriceChart";
import { OrderBook } from "./components/dashboard/OrderBook";
import { useArena } from "./hooks/useArena";
import { Play, Square, RotateCcw, Activity } from "lucide-react";
import { listen } from "@tauri-apps/api/event";

function App() {
  const { data: arenaData, history, isRunning, start, stop } = useArena();
  const [logs, setLogs] = useState<string[]>([]);

  // Listen for logs (if they happen to be in the event, though Rust side needs update for this)
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
    // Explicit background colors help debug "black screen" issues
    <div className="min-h-screen bg-slate-950 text-slate-50 p-4 font-sans selection:bg-indigo-500/30">
      {/* Header */}
      <header className="flex items-center justify-between mb-6 border-b border-slate-800 pb-4">
        <div className="flex items-center gap-3">
          <Activity className="text-indigo-400 w-8 h-8" />
          <h1 className="text-2xl font-bold tracking-tight">nglab <span className="text-indigo-400">Arena</span></h1>
        </div>

        <div className="flex gap-2">
          {!isRunning ? (
            <button
              onClick={handleStart}
              className="flex items-center gap-2 bg-indigo-600 hover:bg-indigo-500 px-4 py-2 rounded-lg transition-colors font-medium shadow-lg shadow-indigo-500/20"
            >
              <Play size={18} fill="currentColor" /> Start Simulation
            </button>
          ) : (
            <button
              onClick={handleStop}
              className="flex items-center gap-2 bg-rose-600 hover:bg-rose-500 px-4 py-2 rounded-lg transition-colors font-medium"
            >
              <Square size={18} fill="currentColor" /> Stop
            </button>
          )}
          <button className="p-2 bg-slate-800 hover:bg-slate-700 rounded-lg transition-colors border border-slate-700">
            <RotateCcw size={20} />
          </button>
        </div>
      </header>

      {/* Main Dashboard Layout */}
      <main className="grid grid-cols-12 gap-6 h-[calc(100vh-120px)]">

        {/* Left: Charts & Orderbook */}
        <div className="col-span-8 flex flex-col gap-6 overflow-hidden">
          <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-4 flex-1 relative overflow-hidden">
            <h3 className="text-sm font-semibold text-slate-400 mb-2 uppercase tracking-wider">Live Price (Polymarket)</h3>
            <div className="h-full w-full bg-slate-950/50 rounded-lg">
              {/* Chart component needs a height-constrained parent */}
              <PriceChart data={history} />
            </div>
          </div>

          <div className="h-64 bg-slate-900/50 border border-slate-800 rounded-xl p-4">
            <h3 className="text-sm font-semibold text-slate-400 mb-2 uppercase tracking-wider">Order Book Depth</h3>
            <OrderBook book={arenaData?.orderbook || { bids: {}, asks: {}, timestamp: 0 }} />
          </div>
        </div>

        {/* Right: Stats & Logs */}
        <div className="col-span-4 flex flex-col gap-6 overflow-hidden">
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
      </main>
    </div>
  );
}

export default App;