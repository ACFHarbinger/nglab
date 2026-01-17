import { useState } from "react";
import clsx from "clsx";
import { Wallet, Info } from "lucide-react";

interface TradingFormProps {
  symbol: string;
  currentPrice: number;
}

export function TradingFormWidget({ symbol, currentPrice }: TradingFormProps) {
  const [side, setSide] = useState<"buy" | "sell">("buy");
  const [amount, setAmount] = useState<string>("");

  const maxBuy = 1000; // Mock wallet balance
  const maxSell = 500; // Mock position size

  const parsedAmount = parseFloat(amount) || 0;
  const estimatedShares = parsedAmount / (currentPrice || 1);

  // Simple fee calc (mock)
  const fee = parsedAmount * 0.001;

  return (
    <div className="flex flex-col h-full bg-slate-900 border-l border-slate-800">
      {/* Buy/Sell Tabs */}
      <div className="flex border-b border-slate-800">
        <button
          onClick={() => setSide("buy")}
          className={clsx(
            "flex-1 py-3 text-sm font-bold uppercase tracking-wide transition-colors",
            side === "buy"
              ? "bg-emerald-600/10 text-emerald-400 border-b-2 border-emerald-500"
              : "text-slate-500 hover:text-slate-300 hover:bg-slate-800/50",
          )}
        >
          Buy
        </button>
        <button
          onClick={() => setSide("sell")}
          className={clsx(
            "flex-1 py-3 text-sm font-bold uppercase tracking-wide transition-colors",
            side === "sell"
              ? "bg-rose-600/10 text-rose-400 border-b-2 border-rose-500"
              : "text-slate-500 hover:text-slate-300 hover:bg-slate-800/50",
          )}
        >
          Sell
        </button>
      </div>

      <div className="p-4 flex flex-col gap-4 flex-1">
        {/* Inputs */}
        <div className="space-y-1">
          <div className="flex justify-between text-xs text-slate-400">
            <span>Limit Price</span>
            <span className="text-indigo-400 cursor-pointer hover:underline">
              Market
            </span>
          </div>
          <div className="relative">
            <input
              type="number"
              className="w-full bg-slate-950 border border-slate-700 rounded-lg p-2.5 text-right font-mono text-white focus:outline-none focus:ring-1 focus:ring-indigo-500"
              placeholder="0.00"
              defaultValue={currentPrice.toFixed(3)}
            />
            <span className="absolute left-3 top-2.5 text-slate-500 text-sm">
              USDC
            </span>
          </div>
        </div>

        <div className="space-y-1">
          <div className="flex justify-between text-xs text-slate-400">
            <span>Amount</span>
            <div className="flex items-center gap-1">
              <Wallet size={10} />
              <span>
                {side === "buy"
                  ? `${maxBuy.toFixed(2)} USDC`
                  : `${maxSell} Shares`}
              </span>
            </div>
          </div>
          <div className="relative">
            <input
              type="number"
              value={amount}
              onChange={(e) => setAmount(e.target.value)}
              className="w-full bg-slate-950 border border-slate-700 rounded-lg p-2.5 text-right font-mono text-white focus:outline-none focus:ring-1 focus:ring-indigo-500"
              placeholder="0.00"
            />
            <span className="absolute left-3 top-2.5 text-slate-500 text-sm">
              USDC
            </span>
          </div>
        </div>

        {/* Percentage Slider (Visual Only) */}
        <div className="flex gap-1 h-1.5 w-full bg-slate-800 rounded overflow-hidden cursor-pointer">
          <div className="w-1/4 hover:bg-indigo-500/50 bg-slate-700 transition-colors" />
          <div className="w-1/4 hover:bg-indigo-500/50 bg-slate-700 border-l border-slate-900 transition-colors" />
          <div className="w-1/4 hover:bg-indigo-500/50 bg-slate-700 border-l border-slate-900 transition-colors" />
          <div className="w-1/4 hover:bg-indigo-500/50 bg-slate-700 border-l border-slate-900 transition-colors" />
        </div>

        {/* Summary Stats */}
        <div className="bg-slate-950/50 rounded-lg p-3 space-y-2 border border-slate-800/50 mt-2">
          <div className="flex justify-between text-xs">
            <span className="text-slate-500">Est. Shares</span>
            <span className="font-mono text-slate-200">
              {estimatedShares.toFixed(2)}
            </span>
          </div>
          <div className="flex justify-between text-xs">
            <span className="text-slate-500 flex items-center gap-1">
              Fee <Info size={10} />
            </span>
            <span className="font-mono text-slate-200">${fee.toFixed(4)}</span>
          </div>
          <div className="flex justify-between text-xs pt-2 border-t border-slate-800/50">
            <span className="text-slate-400 font-bold">Total</span>
            <span className="font-mono text-white text-sm font-bold">
              ${parsedAmount.toFixed(2)}
            </span>
          </div>
        </div>

        <div className="flex-1" />

        {/* Action Button */}
        <button
          className={clsx(
            "w-full py-3.5 rounded-lg font-bold text-sm tracking-wide shadow-lg transition-all active:scale-[0.98]",
            side === "buy"
              ? "bg-emerald-600 hover:bg-emerald-500 text-white shadow-emerald-900/20"
              : "bg-rose-600 hover:bg-rose-500 text-white shadow-rose-900/20",
          )}
        >
          {side === "buy" ? `Buy ${symbol}` : `Sell ${symbol}`}
        </button>
      </div>
    </div>
  );
}
