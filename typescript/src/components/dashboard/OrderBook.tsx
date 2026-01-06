import { useMemo } from 'react';
import { OrderBook as IOrderBook } from '../../hooks/useArena';

export function OrderBook({ book }: { book: IOrderBook }) {
    const bids = useMemo(() => {
        return Object.values(book.bids || {})
            .sort((a, b) => b.price - a.price)
            .slice(0, 15); // Top 15
    }, [book.bids]);

    const asks = useMemo(() => {
        return Object.values(book.asks || {})
            .sort((a, b) => a.price - b.price)
            .slice(0, 15); // Top 15
    }, [book.asks]);

    const maxVol = useMemo(() => {
        const bidMax = Math.max(...bids.map(b => b.total_quantity), 0);
        const askMax = Math.max(...asks.map(a => a.total_quantity), 0);
        return Math.max(bidMax, askMax) || 1;
    }, [bids, asks]);

    return (
        <div className="flex flex-col h-full bg-gray-950 p-4 rounded-lg border border-gray-800">
            <h2 className="text-sm text-gray-400 mb-2 uppercase tracking-wide font-semibold">Order Book</h2>
            <div className="flex gap-2 flex-1 overflow-hidden font-mono text-xs">
                {/* Bids */}
                <div className="flex-1 flex flex-col">
                    <div className="flex justify-between text-gray-500 mb-1 px-1">
                        <span>Vol</span>
                        <span>Bid</span>
                    </div>
                    <div className="flex-1 overflow-y-auto custom-scrollbar">
                        {bids.map(level => (
                            <div key={level.price} className="relative flex justify-between items-center py-0.5 px-1 hover:bg-gray-900 group">
                                <span className="z-10 text-gray-300">{level.total_quantity.toFixed(2)}</span>
                                <span className="z-10 text-green-400 font-medium">{level.price.toFixed(2)}</span>
                                {/* Depth Bar */}
                                <div
                                    className="absolute right-0 top-0 bottom-0 bg-green-900/20"
                                    style={{ width: `${(level.total_quantity / maxVol) * 100}%` }}
                                />
                            </div>
                        ))}
                    </div>
                </div>

                {/* Asks */}
                <div className="flex-1 flex flex-col">
                    <div className="flex justify-between text-gray-500 mb-1 px-1">
                        <span>Ask</span>
                        <span>Vol</span>
                    </div>
                    <div className="flex-1 overflow-y-auto custom-scrollbar">
                        {asks.map(level => (
                            <div key={level.price} className="relative flex justify-between items-center py-0.5 px-1 hover:bg-gray-900">
                                <span className="z-10 text-red-400 font-medium">{level.price.toFixed(2)}</span>
                                <span className="z-10 text-gray-300">{level.total_quantity.toFixed(2)}</span>
                                {/* Depth Bar */}
                                <div
                                    className="absolute left-0 top-0 bottom-0 bg-red-900/20"
                                    style={{ width: `${(level.total_quantity / maxVol) * 100}%` }}
                                />
                            </div>
                        ))}
                    </div>
                </div>
            </div>
        </div>
    );
}
