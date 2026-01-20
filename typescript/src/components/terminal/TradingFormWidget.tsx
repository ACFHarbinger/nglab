import { useState, useMemo } from "react";
import clsx from "clsx";
import { Wallet, Info, ChevronDown, Layers } from "lucide-react";

interface Outcome {
  id: string;
  name: string;
}

interface TradingFormProps {
  symbol: string;
  currentPrice: number;
  outcomes?: Outcome[];
  livePrices?: Record<string, number>;
}

// Generate distinct colors for multi-outcome markets
const outcomeColors = [
  { bg: "bg-emerald-600", bgHover: "hover:bg-emerald-500", bgLight: "bg-emerald-600/10", text: "text-emerald-400", border: "border-emerald-500" },
  { bg: "bg-rose-600", bgHover: "hover:bg-rose-500", bgLight: "bg-rose-600/10", text: "text-rose-400", border: "border-rose-500" },
  { bg: "bg-blue-600", bgHover: "hover:bg-blue-500", bgLight: "bg-blue-600/10", text: "text-blue-400", border: "border-blue-500" },
  { bg: "bg-amber-600", bgHover: "hover:bg-amber-500", bgLight: "bg-amber-600/10", text: "text-amber-400", border: "border-amber-500" },
  { bg: "bg-purple-600", bgHover: "hover:bg-purple-500", bgLight: "bg-purple-600/10", text: "text-purple-400", border: "border-purple-500" },
  { bg: "bg-cyan-600", bgHover: "hover:bg-cyan-500", bgLight: "bg-cyan-600/10", text: "text-cyan-400", border: "border-cyan-500" },
  { bg: "bg-pink-600", bgHover: "hover:bg-pink-500", bgLight: "bg-pink-600/10", text: "text-pink-400", border: "border-pink-500" },
  { bg: "bg-orange-600", bgHover: "hover:bg-orange-500", bgLight: "bg-orange-600/10", text: "text-orange-400", border: "border-orange-500" },
];

export function TradingFormWidget({ currentPrice, outcomes, livePrices }: TradingFormProps) {
  // Default to Yes/No if no outcomes provided
  const marketOutcomes = useMemo(() => {
    if (outcomes && outcomes.length > 0) return outcomes;
    return [
      { id: "yes", name: "Yes" },
      { id: "no", name: "No" },
    ];
  }, [outcomes]);

  const isMultiOutcome = marketOutcomes.length > 2;
  const [side, setSide] = useState<"buy" | "sell">("buy");
  const [selectedOutcomeIdx, setSelectedOutcomeIdx] = useState(0);
  const [amount, setAmount] = useState<string>("");
  const [showOutcomeDropdown, setShowOutcomeDropdown] = useState(false);

  const maxBuy = 1000; // Mock wallet balance
  const maxSell = 500; // Mock position size

  const selectedOutcome = marketOutcomes[selectedOutcomeIdx];
  const colorScheme = outcomeColors[selectedOutcomeIdx % outcomeColors.length];

  // Get price for selected outcome
  const getOutcomePrice = (outcomeId: string, idx: number): number => {
    if (livePrices && livePrices[outcomeId] !== undefined) {
      return livePrices[outcomeId];
    }
    // For binary markets, calculate complementary price
    if (marketOutcomes.length === 2) {
      return idx === 0 ? currentPrice : Math.max(0, 1 - currentPrice);
    }
    // For multi-outcome, use equal distribution as fallback
    return 1 / marketOutcomes.length;
  };

  const activePrice = getOutcomePrice(selectedOutcome.id, selectedOutcomeIdx);
  const parsedAmount = parseFloat(amount) || 0;
  const estimatedShares = parsedAmount / (activePrice || 0.01);
  const potentialPayout = estimatedShares * 1; // Each share pays $1 if it wins
  const potentialProfit = potentialPayout - parsedAmount;

  // Simple fee calc (mock)
  const fee = parsedAmount * 0.001;

  return (
    <div className="flex flex-col h-full bg-slate-900 border-l border-slate-800">
      {/* Multi-outcome indicator */}
      {isMultiOutcome && (
        <div className="px-3 py-2 bg-indigo-500/10 border-b border-indigo-500/20 flex items-center gap-2">
          <Layers size={14} className="text-indigo-400" />
          <span className="text-xs text-indigo-300">
            Multi-outcome market ({marketOutcomes.length} options)
          </span>
        </div>
      )}

      {/* Outcome Selector */}
      {isMultiOutcome ? (
        <div className="p-2 border-b border-slate-800">
          <div className="relative">
            <button
              onClick={() => setShowOutcomeDropdown(!showOutcomeDropdown)}
              className={clsx(
                "w-full flex items-center justify-between px-3 py-2.5 rounded-lg text-sm font-medium transition-colors",
                colorScheme.bg,
                colorScheme.bgHover,
                "text-white"
              )}
            >
              <span className="flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-white/30" />
                {selectedOutcome.name}
              </span>
              <div className="flex items-center gap-2">
                <span className="font-mono text-xs opacity-75">
                  ${activePrice.toFixed(3)}
                </span>
                <ChevronDown
                  size={14}
                  className={clsx(
                    "transition-transform",
                    showOutcomeDropdown && "rotate-180"
                  )}
                />
              </div>
            </button>

            {showOutcomeDropdown && (
              <div className="absolute z-20 top-full left-0 right-0 mt-1 bg-slate-800 border border-slate-700 rounded-lg shadow-xl overflow-hidden max-h-60 overflow-y-auto">
                {marketOutcomes.map((outcome, idx) => {
                  const price = getOutcomePrice(outcome.id, idx);
                  const colors = outcomeColors[idx % outcomeColors.length];
                  return (
                    <button
                      key={outcome.id}
                      onClick={() => {
                        setSelectedOutcomeIdx(idx);
                        setShowOutcomeDropdown(false);
                      }}
                      className={clsx(
                        "w-full flex items-center justify-between px-3 py-2.5 text-sm transition-colors",
                        selectedOutcomeIdx === idx
                          ? `${colors.bgLight} ${colors.text}`
                          : "text-slate-300 hover:bg-slate-700"
                      )}
                    >
                      <span className="flex items-center gap-2">
                        <span
                          className={clsx(
                            "w-2 h-2 rounded-full",
                            colors.bg
                          )}
                        />
                        {outcome.name}
                      </span>
                      <span className="font-mono text-xs">
                        ${price.toFixed(3)}
                      </span>
                    </button>
                  );
                })}
              </div>
            )}
          </div>
        </div>
      ) : (
        /* Binary Outcome Toggle */
        <div className="grid grid-cols-2 gap-1 p-2 border-b border-slate-800">
          {marketOutcomes.map((outcome, idx) => {
            const colors = outcomeColors[idx % outcomeColors.length];
            return (
              <button
                key={outcome.id}
                onClick={() => setSelectedOutcomeIdx(idx)}
                className={clsx(
                  "py-1.5 text-xs font-bold uppercase tracking-wider rounded transition-colors",
                  selectedOutcomeIdx === idx
                    ? `${colors.bg} text-white`
                    : "bg-slate-800 text-slate-500 hover:bg-slate-700"
                )}
              >
                {outcome.name}
              </button>
            );
          })}
        </div>
      )}

      {/* Buy/Sell Tabs */}
      <div className="flex border-b border-slate-800">
        <button
          onClick={() => setSide("buy")}
          className={clsx(
            "flex-1 py-3 text-sm font-bold uppercase tracking-wide transition-colors",
            side === "buy"
              ? "bg-emerald-600/10 text-emerald-400 border-b-2 border-emerald-500"
              : "text-slate-500 hover:text-slate-300 hover:bg-slate-800/50"
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
              : "text-slate-500 hover:text-slate-300 hover:bg-slate-800/50"
          )}
        >
          Sell
        </button>
      </div>

      <div className="p-4 flex flex-col gap-4 flex-1">
        {/* Current outcome price display */}
        <div className="flex items-center justify-between text-xs">
          <span className="text-slate-400">
            {selectedOutcome.name} Price
          </span>
          <span className={clsx("font-mono font-bold", colorScheme.text)}>
            ${activePrice.toFixed(3)}
          </span>
        </div>

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
              key={`${selectedOutcome.id}-price`}
              type="number"
              className="w-full bg-slate-950 border border-slate-700 rounded-lg p-2.5 text-right font-mono text-white focus:outline-none focus:ring-1 focus:ring-indigo-500"
              placeholder="0.00"
              defaultValue={activePrice.toFixed(3)}
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
          {[25, 50, 75, 100].map((pct) => (
            <div
              key={pct}
              onClick={() => setAmount(((maxBuy * pct) / 100).toFixed(2))}
              className="flex-1 hover:bg-indigo-500/50 bg-slate-700 border-l border-slate-900 first:border-l-0 transition-colors"
              title={`${pct}%`}
            />
          ))}
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
            <span className="text-slate-500">Potential Payout</span>
            <span className="font-mono text-emerald-400">
              ${potentialPayout.toFixed(2)}
            </span>
          </div>
          <div className="flex justify-between text-xs">
            <span className="text-slate-500">Potential Profit</span>
            <span className={clsx(
              "font-mono",
              potentialProfit >= 0 ? "text-emerald-400" : "text-rose-400"
            )}>
              {potentialProfit >= 0 ? "+" : ""}${potentialProfit.toFixed(2)}
            </span>
          </div>
          <div className="flex justify-between text-xs">
            <span className="text-slate-500 flex items-center gap-1">
              Fee <Info size={10} />
            </span>
            <span className="font-mono text-slate-200">${fee.toFixed(4)}</span>
          </div>
          <div className="flex justify-between text-xs pt-2 border-t border-slate-800/50">
            <span className="text-slate-400 font-bold">Total Cost</span>
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
              ? `${colorScheme.bg} ${colorScheme.bgHover} text-white shadow-emerald-900/20`
              : "bg-rose-600 hover:bg-rose-500 text-white shadow-rose-900/20"
          )}
        >
          {side === "buy" ? `Buy ${selectedOutcome.name}` : `Sell ${selectedOutcome.name}`}
        </button>

        {/* Outcome probabilities summary for multi-outcome */}
        {isMultiOutcome && (
          <div className="mt-2 pt-2 border-t border-slate-800">
            <div className="text-xs text-slate-500 mb-2">All Outcomes</div>
            <div className="space-y-1">
              {marketOutcomes.map((outcome, idx) => {
                const price = getOutcomePrice(outcome.id, idx);
                const colors = outcomeColors[idx % outcomeColors.length];
                const probability = (price * 100).toFixed(1);
                return (
                  <div
                    key={outcome.id}
                    className="flex items-center gap-2 text-xs"
                  >
                    <span
                      className={clsx("w-1.5 h-1.5 rounded-full", colors.bg)}
                    />
                    <span className="text-slate-400 flex-1 truncate">
                      {outcome.name}
                    </span>
                    <span className="font-mono text-slate-300">
                      {probability}%
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
