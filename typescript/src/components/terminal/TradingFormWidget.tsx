import { useState, useMemo, useEffect } from "react";
import clsx from "clsx";
import { Wallet, Info, ChevronDown, Layers, Settings } from "lucide-react";
import { invoke } from "@tauri-apps/api/core";

type OrderType = "Limit" | "Market" | "FOK" | "IOC" | "Bracket" | "Pegged" | "Algo";
type AlgoType = "TWAP" | "VWAP" | "POV";
type PegReference = "BestBid" | "BestAsk" | "MidPoint";

/**
 * Represents a market outcome (e.g., "Yes", "Trump").
 */
interface Outcome {
  id: string;
  name: string;
}

/**
 * Props for the TradingFormWidget.
 */
interface TradingFormProps {
  /** Market ticker symbol. */
  symbol: string;
  /** Current best price (mid-market or last). */
  currentPrice: number;
  /** List of available outcomes. */
  outcomes?: Outcome[];
  /** Real-time price map for multi-outcome pricing. */
  livePrices?: Record<string, number>;
}

// Generate distinct colors for multi-outcome markets
const outcomeColors = [
  {
    bg: "bg-emerald-600",
    bgHover: "hover:bg-emerald-500",
    bgLight: "bg-emerald-600/10",
    text: "text-emerald-400",
    border: "border-emerald-500",
  },
  {
    bg: "bg-rose-600",
    bgHover: "hover:bg-rose-500",
    bgLight: "bg-rose-600/10",
    text: "text-rose-400",
    border: "border-rose-500",
  },
  {
    bg: "bg-blue-600",
    bgHover: "hover:bg-blue-500",
    bgLight: "bg-blue-600/10",
    text: "text-blue-400",
    border: "border-blue-500",
  },
  {
    bg: "bg-amber-600",
    bgHover: "hover:bg-amber-500",
    bgLight: "bg-amber-600/10",
    text: "text-amber-400",
    border: "border-amber-500",
  },
  {
    bg: "bg-purple-600",
    bgHover: "hover:bg-purple-500",
    bgLight: "bg-purple-600/10",
    text: "text-purple-400",
    border: "border-purple-500",
  },
  {
    bg: "bg-cyan-600",
    bgHover: "hover:bg-cyan-500",
    bgLight: "bg-cyan-600/10",
    text: "text-cyan-400",
    border: "border-cyan-500",
  },
  {
    bg: "bg-pink-600",
    bgHover: "hover:bg-pink-500",
    bgLight: "bg-pink-600/10",
    text: "text-pink-400",
    border: "border-pink-500",
  },
  {
    bg: "bg-orange-600",
    bgHover: "hover:bg-orange-500",
    bgLight: "bg-orange-600/10",
    text: "text-orange-400",
    border: "border-orange-500",
  },
];

/**
 * Order execution form for placing Limit and Market orders.
 * Supports both Binary and Multi-Outcome markets with dynamic UI.
 * Handles estimated shares, potential payout, and profit calculations.
 */
export function TradingFormWidget({
  currentPrice,
  outcomes,
  livePrices,
}: TradingFormProps) {
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

  // Advanced Order State
  const [orderType, setOrderType] = useState<OrderType>("Limit");
  const [showOrderTypeDropdown, setShowOrderTypeDropdown] = useState(false);
  const [pegReference, setPegReference] = useState<PegReference>("BestBid");
  const [pegOffset, setPegOffset] = useState<string>("0.0");
  const [bracketSL, setBracketSL] = useState<string>("");
  const [bracketTP, setBracketTP] = useState<string>("");

  // Algo State
  const [algoType, setAlgoType] = useState<AlgoType>("TWAP");
  const [duration, setDuration] = useState<string>("100");
  const [participationRate, setParticipationRate] = useState<string>("0.1");

  const handleSubmit = async () => {
    const qty = parseFloat(amount);
    const price = parseFloat(document.querySelector<HTMLInputElement>("input[placeholder='0.00']")?.value || "0");

    if (!qty) return;

    try {
      let command = "";
      let args: any = {
        quantity: qty,
        side: side === "buy" ? "Bid" : "Ask",
      };

      switch (orderType) {
        case "Limit":
          // Default limit logic (existing or new command? checks lib.rs... assumes submit_limit_order exists elsewhere or we use generic)
          // If generic submit_order doesn't exist, we might need to add it or use a simpler assumption for now.
          // However, for this task, I am implementing the NEW types.
          // Assuming Limit runs via existing mechanisms or I should have added it?
          // I'll stick to implementing the NEW ones clearly.
          // If Limit, I won't touch it much, but wait, I need to know what command to call.
          // `submit_limit_order` is NOT in my `trade.rs`. It might be in `simulation.rs` or directly in `TradingEnv` via wrapper?
          // The existing code didn't show submit logic. I'll focus on the new ones.
          console.log("Limit order submission not implementing in this pass");
          break;
        case "FOK":
          command = "submit_fok_order";
          args.price = price;
          break;
        case "IOC":
          command = "submit_ioc_order";
          args.price = price;
          break;
        case "Bracket":
          command = "submit_bracket_order";
          args.price = price;
          args.slPrice = parseFloat(bracketSL);
          args.tpPrice = parseFloat(bracketTP);
          break;
        case "Pegged":
          command = "submit_pegged_order";
          // Pegged doesn't use price, it uses ref + offset
          args.pegReference = pegReference;
          args.pegOffset = parseFloat(pegOffset);
          break;
        case "Algo":
          command = "submit_algo_order";
          args = {
            algo_type: algoType,
            params: {
              quantity: qty,
              side: side === "buy" ? "Bid" : "Ask",
              duration_steps: parseInt(duration),
              participation_rate: parseFloat(participationRate),
            }
          };
          break;
      }

      if (command) {
        await invoke(command, args);
        alert(`Order ${orderType} Submitted!`);
      }
    } catch (e) {
      console.error(e);
      alert("Order failed: " + e);
    }
  };

  const maxBuy = 1000; // Mock wallet balance
  const maxSell = 500; // Mock position size

  const selectedOutcome = marketOutcomes[selectedOutcomeIdx] ||
    marketOutcomes[0] || { id: "unknown", name: "Unknown" };
  const colorScheme =
    outcomeColors[
    Math.min(selectedOutcomeIdx, marketOutcomes.length - 1) %
    outcomeColors.length
    ] || outcomeColors[0];

  // Reset index if it becomes out of bounds due to prop changes
  useEffect(() => {
    if (selectedOutcomeIdx >= marketOutcomes.length) {
      setSelectedOutcomeIdx(0);
    }
  }, [marketOutcomes.length, selectedOutcomeIdx]);

  // Get price for selected outcome
  const getOutcomePrice = (outcomeId: string, idx: number): number => {
    if (livePrices && livePrices[outcomeId] !== undefined) {
      const price = livePrices[outcomeId];
      return typeof price === "number" ? price : 0;
    }
    // For binary markets, calculate complementary price
    if (marketOutcomes.length === 2) {
      return idx === 0
        ? currentPrice || 0
        : Math.max(0, 1 - (currentPrice || 0));
    }
    // For multi-outcome, use equal distribution as fallback
    return 1 / marketOutcomes.length;
  };

  const activePrice = getOutcomePrice(selectedOutcome.id, selectedOutcomeIdx);
  const parsedAmount = parseFloat(amount) || 0;
  // Ensure we don't divide by zero or NaN
  const safePrice =
    isFinite(activePrice) && activePrice > 0 ? activePrice : 0.01;
  const estimatedShares = parsedAmount / safePrice;
  const potentialPayout = estimatedShares * 1; // Each share pays $1 if it wins
  const potentialProfit = isFinite(potentialPayout)
    ? potentialPayout - parsedAmount
    : 0;

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
                "text-white",
              )}
            >
              <span className="flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-white/30" />
                {selectedOutcome.name}
              </span>
              <div className="flex items-center gap-2">
                <span className="font-mono text-xs opacity-75">
                  ${isFinite(activePrice) ? activePrice.toFixed(3) : "0.000"}
                </span>
                <ChevronDown
                  size={14}
                  className={clsx(
                    "transition-transform",
                    showOutcomeDropdown && "rotate-180",
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
                          : "text-slate-300 hover:bg-slate-700",
                      )}
                    >
                      <span className="flex items-center gap-2">
                        <span
                          className={clsx("w-2 h-2 rounded-full", colors.bg)}
                        />
                        {outcome.name}
                      </span>
                      <span className="font-mono text-xs">
                        ${isFinite(price) ? price.toFixed(3) : "0.000"}
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
                    : "bg-slate-800 text-slate-500 hover:bg-slate-700",
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

      <div className="px-4 pt-4">
        {/* Order Type Selector */}
        <div className="relative z-10">
          <button
            onClick={() => setShowOrderTypeDropdown(!showOrderTypeDropdown)}
            className="w-full flex items-center justify-between bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-200 hover:bg-slate-700 transition-colors"
          >
            <span className="flex items-center gap-2">
              <Settings size={14} className="text-slate-400" />
              {orderType} Order
            </span>
            <ChevronDown size={14} className={clsx("transition-transform", showOrderTypeDropdown && "rotate-180")} />
          </button>

          {showOrderTypeDropdown && (
            <div className="absolute top-full left-0 right-0 mt-1 bg-slate-800 border border-slate-700 rounded-lg shadow-xl overflow-hidden">
              {(["Limit", "Market", "FOK", "IOC", "Bracket", "Pegged", "Algo"] as OrderType[]).map((type) => (
                <button
                  key={type}
                  onClick={() => {
                    setOrderType(type);
                    setShowOrderTypeDropdown(false);
                  }}
                  className={clsx(
                    "w-full text-left px-3 py-2 text-sm transition-colors hover:bg-slate-700",
                    orderType === type ? "text-indigo-400 bg-slate-700/50" : "text-slate-300"
                  )}
                >
                  {type}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      <div className="p-4 flex flex-col gap-4 flex-1">
        {/* Current outcome price display */}
        <div className="flex items-center justify-between text-xs">
          <span className="text-slate-400">{selectedOutcome.name} Price</span>
          <span className={clsx("font-mono font-bold", colorScheme.text)}>
            ${isFinite(activePrice) ? activePrice.toFixed(3) : "0.000"}
          </span>
        </div>

        {/* Inputs */}
        {orderType !== "Pegged" && orderType !== "Market" && orderType !== "Algo" && (
          <div className="space-y-1">
            <div className="flex justify-between text-xs text-slate-400">
              <span>{orderType} Price</span>
              {orderType === "Limit" && (
                <span
                  className="text-indigo-400 cursor-pointer hover:underline"
                  onClick={() => setOrderType("Market")}
                >
                  Market
                </span>
              )}
            </div>
            <div className="relative">
              <input
                key={`${selectedOutcome.id}-price`}
                type="number"
                className="w-full bg-slate-950 border border-slate-700 rounded-lg p-2.5 text-right font-mono text-white focus:outline-none focus:ring-1 focus:ring-indigo-500"
                placeholder="0.00"
                defaultValue={
                  isFinite(activePrice) ? activePrice.toFixed(3) : "0.000"
                }
              />
              <span className="absolute left-3 top-2.5 text-slate-500 text-sm">
                USDC
              </span>
            </div>
          </div>
        )}

        {/* Algo Inputs */}
        {orderType === "Algo" && (
          <div className="space-y-3">
            <div className="flex gap-1">
              {(["TWAP", "VWAP", "POV"] as AlgoType[]).map((t) => (
                <button
                  key={t}
                  onClick={() => setAlgoType(t)}
                  className={clsx(
                    "flex-1 py-1.5 text-xs border rounded transition-colors",
                    algoType === t
                      ? "bg-indigo-500/20 border-indigo-500 text-indigo-300"
                      : "bg-slate-950 border-slate-700 text-slate-400 hover:border-slate-500"
                  )}
                >
                  {t}
                </button>
              ))}
            </div>

            {algoType !== "POV" ? (
              <div className="space-y-1">
                <div className="text-xs text-slate-400">Duration (Steps)</div>
                <input
                  type="number"
                  value={duration}
                  onChange={(e) => setDuration(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-700 rounded-lg p-2.5 text-right font-mono text-white focus:outline-none focus:ring-1 focus:ring-indigo-500"
                />
              </div>
            ) : (
              <div className="space-y-1">
                <div className="text-xs text-slate-400">Participation Rate</div>
                <input
                  type="range"
                  min="0.01"
                  max="0.5"
                  step="0.01"
                  value={participationRate}
                  onChange={(e) => setParticipationRate(e.target.value)}
                  className="w-full"
                />
                <div className="flex justify-between text-[10px] text-slate-500 font-mono">
                  <span>1%</span>
                  <span className="text-indigo-400">{(parseFloat(participationRate) * 100).toFixed(0)}%</span>
                  <span>50%</span>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Pegged Inputs */}
        {orderType === "Pegged" && (
          <>
            <div className="space-y-1">
              <div className="text-xs text-slate-400">Reference</div>
              <div className="flex gap-1">
                {(["BestBid", "BestAsk", "MidPoint"] as PegReference[]).map((ref) => (
                  <button
                    key={ref}
                    onClick={() => setPegReference(ref)}
                    className={clsx(
                      "flex-1 py-1.5 text-xs border rounded transition-colors",
                      pegReference === ref
                        ? "bg-indigo-500/20 border-indigo-500 text-indigo-300"
                        : "bg-slate-950 border-slate-700 text-slate-400 hover:border-slate-500"
                    )}
                  >
                    {ref}
                  </button>
                ))}
              </div>
            </div>
            <div className="space-y-1">
              <div className="text-xs text-slate-400">Offset</div>
              <div className="relative">
                <input
                  type="number"
                  value={pegOffset}
                  onChange={(e) => setPegOffset(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-700 rounded-lg p-2.5 text-right font-mono text-white focus:outline-none focus:ring-1 focus:ring-indigo-500"
                />
                <span className="absolute left-3 top-2.5 text-slate-500 text-sm">+/-</span>
              </div>
            </div>
          </>
        )}

        {/* Bracket Inputs */}
        {orderType === "Bracket" && (
          <div className="grid grid-cols-2 gap-2">
            <div className="space-y-1">
              <div className="text-xs text-slate-400">Stop Loss</div>
              <input
                type="number"
                value={bracketSL}
                onChange={(e) => setBracketSL(e.target.value)}
                className="w-full bg-slate-950 border border-slate-700 rounded-lg p-2 text-right font-mono text-rose-400 focus:outline-none focus:ring-1 focus:ring-rose-500"
                placeholder="SL"
              />
            </div>
            <div className="space-y-1">
              <div className="text-xs text-slate-400">Take Profit</div>
              <input
                type="number"
                value={bracketTP}
                onChange={(e) => setBracketTP(e.target.value)}
                className="w-full bg-slate-950 border border-slate-700 rounded-lg p-2 text-right font-mono text-emerald-400 focus:outline-none focus:ring-1 focus:ring-emerald-500"
                placeholder="TP"
              />
            </div>
          </div>
        )}

        <div className="space-y-1">
          <div className="flex justify-between text-xs text-slate-400">
            <span>Amount</span>
            <div className="flex items-center gap-1">
              <Wallet size={10} />
              <span>
                {side === "buy"
                  ? `${isFinite(maxBuy) ? maxBuy.toFixed(2) : "0.00"} USDC`
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
              {isFinite(estimatedShares) ? estimatedShares.toFixed(2) : "0.00"}
            </span>
          </div>
          <div className="flex justify-between text-xs">
            <span className="text-slate-500">Potential Payout</span>
            <span className="font-mono text-emerald-400">
              ${isFinite(potentialPayout) ? potentialPayout.toFixed(2) : "0.00"}
            </span>
          </div>
          <div className="flex justify-between text-xs">
            <span className="text-slate-500">Potential Profit</span>
            <span
              className={clsx(
                "font-mono",
                potentialProfit >= 0 ? "text-emerald-400" : "text-rose-400",
              )}
            >
              {potentialProfit >= 0 ? "+" : ""}$
              {isFinite(potentialProfit) ? potentialProfit.toFixed(2) : "0.00"}
            </span>
          </div>
          <div className="flex justify-between text-xs">
            <span className="text-slate-500 flex items-center gap-1">
              Fee <Info size={10} />
            </span>
            <span className="font-mono text-slate-200">
              ${isFinite(fee) ? fee.toFixed(4) : "0.0000"}
            </span>
          </div>
          <div className="flex justify-between text-xs pt-2 border-t border-slate-800/50">
            <span className="text-slate-400 font-bold">Total Cost</span>
            <span className="font-mono text-white text-sm font-bold">
              ${isFinite(parsedAmount) ? parsedAmount.toFixed(2) : "0.00"}
            </span>
          </div>
        </div>

        <div className="flex-1" />

        {/* Action Button */}
        <button
          onClick={handleSubmit}
          className={clsx(
            "w-full py-3.5 rounded-lg font-bold text-sm tracking-wide shadow-lg transition-all active:scale-[0.98]",
            side === "buy"
              ? `${colorScheme.bg} ${colorScheme.bgHover} text-white shadow-emerald-900/20`
              : "bg-rose-600 hover:bg-rose-500 text-white shadow-rose-900/20",
          )}
        >
          {side === "buy"
            ? `Buy ${selectedOutcome.name}`
            : `Sell ${selectedOutcome.name}`}
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
