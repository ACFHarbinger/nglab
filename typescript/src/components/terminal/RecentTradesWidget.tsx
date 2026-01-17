import { useMemo } from 'react';
import clsx from 'clsx';

export interface Trade {
    id: string;
    price: number;
    amount: number; // Quote currency amount (USD)
    size: number;   // Base currency size (Contracts)
    side: 'buy' | 'sell';
    timestamp: number; // Unix timestamp in seconds
}

interface RecentTradesProps {
    trades: Trade[];
}

export function RecentTradesWidget({ trades }: RecentTradesProps) {
    // Sort trades descending by time (newest first)
    const sortedTrades = useMemo(() => {
        return [...trades].sort((a, b) => b.timestamp - a.timestamp);
    }, [trades]);

    return (
        <div className="flex flex-col h-full bg-slate-900 border-l border-slate-800 w-full">
            <div className="p-3 border-b border-slate-800">
                <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Recent Trades</h3>
            </div>

            <div className="flex-1 flex flex-col overflow-hidden text-xs font-mono">
                {/* Header */}
                <div className="flex justify-between px-3 py-2 text-slate-500 bg-slate-950/30">
                    <span className="w-16">Price</span>
                    <span className="w-16 text-right">Size</span>
                    <span className="flex-1 text-right">Time</span>
                </div>

                {/* Trade List */}
                <div className="flex-1 overflow-y-auto custom-scrollbar">
                    {sortedTrades.map((trade) => (
                        <div key={trade.id} className="flex justify-between items-center py-1 px-3 hover:bg-slate-800/50 transition-colors">
                            <span className={clsx("w-16 font-medium", trade.side === 'buy' ? "text-emerald-400" : "text-rose-400")}>
                                {trade.price.toFixed(3)}
                            </span>
                            <span className="w-16 text-right text-slate-300">
                                {trade.size.toFixed(0)}
                            </span>
                            <span className="flex-1 text-right text-slate-500">
                                {new Date(trade.timestamp * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
                            </span>
                        </div>
                    ))}
                    {sortedTrades.length === 0 && (
                        <div className="p-4 text-center text-slate-600 italic">No recent trades</div>
                    )}
                </div>
            </div>
        </div>
    );
}
