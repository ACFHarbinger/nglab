import { useMemo } from "react";

// Reusing types/logic from the main dashboard but styling for the terminal
interface OrderBookLevel {
  price: number;
  total_quantity: number;
}

interface OrderBookProps {
  bids: Record<string, OrderBookLevel>;
  asks: Record<string, OrderBookLevel>;
}

export function OrderBookWidget({ bids, asks }: OrderBookProps) {
  const sortedBids = useMemo(() => {
    return Object.values(bids)
      .sort((a, b) => b.price - a.price)
      .slice(0, 15);
  }, [bids]);

  const sortedAsks = useMemo(() => {
    return Object.values(asks)
      .sort((a, b) => a.price - b.price)
      .slice(0, 15);
  }, [asks]);

  const maxVol = useMemo(() => {
    const bidMax = Math.max(...sortedBids.map((b) => b.total_quantity), 0);
    const askMax = Math.max(...sortedAsks.map((a) => a.total_quantity), 0);
    return Math.max(bidMax, askMax) || 1;
  }, [sortedBids, sortedAsks]);

  return (
    <div className="flex flex-col h-full bg-slate-900 border-l border-slate-800 w-72">
      <div className="p-3 border-b border-slate-800">
        <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
          Order Book
        </h3>
      </div>

      <div className="flex-1 flex flex-col overflow-hidden text-xs font-mono">
        {/* Header */}
        <div className="flex justify-between px-2 py-1 text-slate-500 bg-slate-950/30">
          <span>Price</span>
          <span>Size</span>
          <span>Total</span>
        </div>

        {/* Asks (Red, Top) - Reversed for standard vertical layout */}
        <div className="flex-1 overflow-y-auto flex flex-col-reverse custom-scrollbar">
          {sortedAsks.map((ask) => (
            <div
              key={ask.price}
              className="relative flex justify-between items-center py-0.5 px-2 hover:bg-slate-800/50"
            >
              {/* Depth Bar */}
              <div
                className="absolute right-0 top-0 bottom-0 bg-rose-900/20"
                style={{ width: `${(ask.total_quantity / maxVol) * 100}%` }}
              />
              <span className="z-10 text-rose-400">{ask.price.toFixed(3)}</span>
              <span className="z-10 text-slate-300">
                {ask.total_quantity.toFixed(0)}
              </span>
              <span className="z-10 text-slate-500">
                {(ask.price * ask.total_quantity).toFixed(0)}
              </span>
            </div>
          ))}
        </div>

        {/* Spread / Mid Price */}
        <div className="py-2 border-y border-slate-800 bg-slate-950 text-center">
          {sortedAsks.length > 0 && sortedBids.length > 0 ? (
            <span className="text-lg font-bold text-slate-200">
              {((sortedAsks[0].price + sortedBids[0].price) / 2).toFixed(3)}
            </span>
          ) : (
            <span className="text-slate-500">-</span>
          )}
        </div>

        {/* Bids (Green, Bottom) */}
        <div className="flex-1 overflow-y-auto custom-scrollbar">
          {sortedBids.map((bid) => (
            <div
              key={bid.price}
              className="relative flex justify-between items-center py-0.5 px-2 hover:bg-slate-800/50"
            >
              {/* Depth Bar */}
              <div
                className="absolute right-0 top-0 bottom-0 bg-emerald-900/20"
                style={{ width: `${(bid.total_quantity / maxVol) * 100}%` }}
              />
              <span className="z-10 text-emerald-400">
                {bid.price.toFixed(3)}
              </span>
              <span className="z-10 text-slate-300">
                {bid.total_quantity.toFixed(0)}
              </span>
              <span className="z-10 text-slate-500">
                {(bid.price * bid.total_quantity).toFixed(0)}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
