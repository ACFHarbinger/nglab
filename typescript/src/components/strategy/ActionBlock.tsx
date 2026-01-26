
import { Trash } from "lucide-react";

export interface StrategyAction {
    id: string;
    type: "Buy" | "Sell" | "Close";
    quantity: number; // For Buy/Sell, 0 for Close (Close All)
    orderType: "Market" | "Limit";
}

interface ActionBlockProps {
    action: StrategyAction;
    onChange: (a: StrategyAction) => void;
    onRemove: () => void;
}

export function ActionBlock({ action, onChange, onRemove }: ActionBlockProps) {
    return (
        <div className="bg-slate-800/80 rounded p-3 border border-indigo-500/30 flex flex-col gap-2 relative overflow-hidden">
            <div className="absolute left-0 top-0 bottom-0 w-1 bg-indigo-500"></div>

            <div className="flex justify-between items-center pl-2">
                <span className="text-xs font-bold text-indigo-300 uppercase tracking-widest">
                    Execute Action
                </span>
                <button onClick={onRemove} className="text-slate-500 hover:text-rose-400">
                    <Trash size={14} />
                </button>
            </div>

            <div className="flex gap-2 items-center pl-2">
                <select
                    value={action.type}
                    onChange={(e) => onChange({ ...action, type: e.target.value as any })}
                    className="bg-slate-900 border border-slate-700 rounded text-xs px-2 py-1 text-slate-300 focus:border-indigo-500 outline-none font-bold"
                >
                    <option value="Buy">BUY</option>
                    <option value="Sell">SELL</option>
                    <option value="Close">CLOSE POSITION</option>
                </select>

                {action.type !== "Close" && (
                    <>
                        <input
                            type="number"
                            placeholder="Qty"
                            value={action.quantity}
                            onChange={(e) => onChange({ ...action, quantity: parseFloat(e.target.value) })}
                            className="bg-slate-900 border border-slate-700 rounded text-xs px-2 py-1 text-slate-300 focus:border-indigo-500 outline-none w-20"
                        />
                        <select
                            value={action.orderType}
                            onChange={(e) => onChange({ ...action, orderType: e.target.value as any })}
                            className="bg-slate-900 border border-slate-700 rounded text-xs px-2 py-1 text-slate-300 focus:border-indigo-500 outline-none"
                        >
                            <option value="Market">Market</option>
                            <option value="Limit">Limit</option>
                        </select>
                    </>
                )}
            </div>
        </div>
    );
}
