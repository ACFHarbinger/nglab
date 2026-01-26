import { useMemo, useState } from "react";
import { List, BarChart2 } from "lucide-react";
import clsx from "clsx";
import { DepthChart } from "../charts/DepthChart";
import { ImbalanceIndicator } from "./ImbalanceIndicator";

/**
 * Represents a single price level in the order book.
 */
interface OrderBookLevel {
  /** Price level. */
  price: number;
  /** aggregated quantity at this price. */
  total_quantity: number;
}

/**
 * Props for the OrderBookWidget.
 */
interface OrderBookProps {
  /** Map of bid levels by price. */
  bids: Record<string, OrderBookLevel>;
  /** Map of ask levels by price. */
  asks: Record<string, OrderBookLevel>;
}

/**
 * Terminal-optimized vertical Order Book visualization.
 * Displays market depth with visual volume bars and spread calculation.
 */
export function OrderBookWidget({ bids, asks }: OrderBookProps) {
  const [visMode, setVisMode] = useState<"List" | "Depth">("List");
  const sortedBids = useMemo(() => {
    if (!bids) return [];
    return Object.values(bids)
      .filter((b) => b && typeof b.price === "number")
      .sort((a, b) => b.price - a.price)
      .slice(0, 15);
  }, [bids]);

  const sortedAsks = useMemo(() => {
    if (!asks) return [];
    return Object.values(asks)
      .filter((a) => a && typeof a.price === "number")
      .sort((a, b) => a.price - b.price)
      .slice(0, 15);
  }, [asks]);

  const maxVol = useMemo(() => {
    const bidMax = Math.max(...sortedBids.map((b) => b.total_quantity), 0);
    const askMax = Math.max(...sortedAsks.map((a) => a.total_quantity), 0);
    return Math.max(bidMax, askMax) || 1;
  }, [sortedBids, sortedAsks]);

  const totalBidVol = useMemo(() => sortedBids.reduce((acc, b) => acc + b.total_quantity, 0), [sortedBids]);
  const totalAskVol = useMemo(() => sortedAsks.reduce((acc, a) => acc + a.total_quantity, 0), [sortedAsks]);

  return (
    <div className="flex flex-col h-full bg-slate-900 border-l border-slate-800 w-72">
      <div className="p-2 border-b border-slate-800 flex justify-between items-center">
        <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider pl-1">
          Order Book
        </h3>
        <div className="flex bg-slate-950 rounded p-0.5">
          <button
            onClick={() => setVisMode("List")}
            className={clsx("p-1 rounded text-slate-400 hover:text-white transition-colors", visMode === "List" && "bg-slate-800 text-white shadow-sm")}
          >
            <List size={14} />
          </button>
          <button
            onClick={() => setVisMode("Depth")}
            className={clsx("p-1 rounded text-slate-400 hover:text-white transition-colors", visMode === "Depth" && "bg-slate-800 text-white shadow-sm")}
          >
            <BarChart2 size={14} />
          </button>
        </div>
      </div>

      <ImbalanceIndicator bidVolume={totalBidVol} askVolume={totalAskVol} />

      {visMode === "Depth" ? (
        <div className="flex-1 p-2 flex flex-col items-center justify-center bg-slate-950/30">
          <DepthChart bids={sortedBids} asks={sortedAsks} height={200} />
          <div className="mt-4 text-xs text-slate-500 text-center">
            Cumulative Depth Visualization
          </div>
        </div>
      ) : (
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
                <span className="z-10 text-rose-400">
                  {isFinite(ask.price) ? ask.price.toFixed(3) : "0.000"}
                </span>
                <span className="z-10 text-slate-300">
                  {isFinite(ask.total_quantity)
                    ? ask.total_quantity.toFixed(0)
                    : "0"}
                </span>
                <span className="z-10 text-slate-500">
                  {isFinite(ask.price) && isFinite(ask.total_quantity)
                    ? (ask.price * ask.total_quantity).toFixed(0)
                    : "0"}
                </span>
              </div>
            ))}
          </div>

          {/* Spread / Mid Price */}
          <div className="py-2 border-y border-slate-800 bg-slate-950 text-center">
            {sortedAsks.length > 0 && sortedBids.length > 0 ? (
              <span className="text-lg font-bold text-slate-200">
                {isFinite(sortedAsks[0].price) && isFinite(sortedBids[0].price)
                  ? ((sortedAsks[0].price + sortedBids[0].price) / 2).toFixed(3)
                  : "-"}
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
                  {isFinite(bid.price) ? bid.price.toFixed(3) : "0.000"}
                </span>
                <span className="z-10 text-slate-300">
                  {isFinite(bid.total_quantity)
                    ? bid.total_quantity.toFixed(0)
                    : "0"}
                </span>
                <span className="z-10 text-slate-500">
                  {isFinite(bid.price) && isFinite(bid.total_quantity)
                    ? (bid.price * bid.total_quantity).toFixed(0)
                    : "0"}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
