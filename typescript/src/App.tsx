import { useArena } from './hooks/useArena';
import { PriceChart } from './components/charts/PriceChart';
import { OrderBook } from './components/dashboard/OrderBook';
import { Play, Square, RefreshCw } from 'lucide-react';
import clsx from 'clsx';

function App() {
  const { data, history, isRunning, start, stop } = useArena();

  return (
    <div className="h-screen w-screen bg-black text-white flex flex-col font-sans overflow-hidden">
      {/* Header */}
      <header className="h-14 border-b border-gray-800 flex items-center px-6 justify-between bg-gray-950/50 backdrop-blur">
        <div className="flex items-center gap-3">
          <div className="w-3 h-3 rounded-full bg-blue-500 animate-pulse"></div>
          <h1 className="text-xl font-bold bg-gradient-to-r from-blue-400 to-emerald-400 bg-clip-text text-transparent">
            NGLAB <span className="text-gray-600 text-sm font-mono ml-2 font-normal hidden sm:inline">v0.1.0</span>
          </h1>
        </div>
        <div className="flex gap-3">
          {!isRunning ? (
            <button
              onClick={start}
              className="flex items-center gap-2 px-4 py-1.5 bg-emerald-600 hover:bg-emerald-700 rounded-md text-sm font-medium transition-all shadow-lg hover:shadow-emerald-900/20 active:scale-95"
            >
              <Play size={16} fill="currentColor" /> Start Simulation
            </button>
          ) : (
            <button
              onClick={stop}
              className="flex items-center gap-2 px-4 py-1.5 bg-rose-600 hover:bg-rose-700 rounded-md text-sm font-medium transition-all shadow-lg hover:shadow-rose-900/20 active:scale-95"
            >
              <Square size={16} fill="currentColor" /> Stop
            </button>
          )}
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 flex overflow-hidden relative">
        {/* Left: Chart & Stats */}
        <div className="flex-1 flex flex-col p-6 gap-6 overflow-y-auto">
          {/* Top Stats */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <StatCard label="Market Price" value={data?.price.toFixed(2) || '0.00'} />
            <StatCard
              label="Portfolio Value"
              value={data?.portfolio_value.toFixed(2) || '10000.00'}
              color="text-blue-400"
              subValue={data ? `${((data.portfolio_value - 10000) / 100).toFixed(2)}%` : '0.00%'}
              subColor={data && data.portfolio_value >= 10000 ? 'text-green-500' : 'text-red-500'}
            />
            <StatCard label="Current Step" value={data?.step.toString() || '0'} color="text-gray-400" />
            <StatCard label="Status" value={isRunning ? 'Running' : 'Idle'} color={isRunning ? 'text-emerald-400' : 'text-gray-500'} />
          </div>

          {/* Chart */}
          <div className="bg-gray-900/50 rounded-xl p-1 border border-gray-800 flex-1 min-h-[400px] shadow-inner relative overflow-hidden">
            <div className="absolute top-4 left-4 z-10 text-xs text-gray-500 font-mono">Real-time Performance</div>
            <PriceChart data={history} />
          </div>
        </div>

        {/* Right: Order Book */}
        <div className="w-80 border-l border-gray-800 bg-gray-950/80 backdrop-blur-sm p-4 flex flex-col gap-4">
          {data?.orderbook ? (
            <OrderBook book={data.orderbook} />
          ) : (
            <div className="h-full flex flex-col items-center justify-center text-gray-700 gap-4">
              <RefreshCw className={clsx("w-8 h-8", isRunning && "animate-spin")} />
              <span className="text-sm font-medium">Waiting for market data...</span>
            </div>
          )}

          {/* Logs Panel Placeholder */}
          <div className="h-1/3 bg-gray-900/50 rounded-lg border border-gray-800 p-3 overflow-hidden flex flex-col">
            <h3 className="text-xs text-gray-500 uppercase mb-2 font-semibold">System Logs</h3>
            <div className="flex-1 font-mono text-[10px] text-gray-400 overflow-y-auto space-y-1">
              <div className="text-blue-500">[SYSTEM] Dashboard initialized</div>
              {isRunning && <div className="text-green-500">[ARENA] Simulation started</div>}
              {!isRunning && history.length > 0 && <div className="text-red-500">[ARENA] Simulation paused</div>}
              {history.slice(-5).reverse().map(h => (
                <div key={h.step} className="opacity-70">Step {h.step}: Price {h.price.toFixed(2)}</div>
              ))}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

function StatCard({ label, value, color = 'text-white', subValue, subColor }: { label: string, value: string, color?: string, subValue?: string, subColor?: string }) {
  return (
    <div className="bg-gray-900/60 border border-gray-800 p-4 rounded-xl shadow-sm hover:border-gray-700 transition-colors">
      <div className="text-xs text-gray-500 uppercase tracking-wider mb-1 font-semibold">{label}</div>
      <div className="flex items-baseline gap-2">
        <div className={`text-2xl font-mono font-medium ${color}`}>{value}</div>
        {subValue && <div className={`text-xs font-mono ${subColor}`}>{subValue}</div>}
      </div>
    </div>
  );
}

export default App;
