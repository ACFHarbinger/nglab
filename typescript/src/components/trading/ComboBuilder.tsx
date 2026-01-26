
import { useState } from "react";
import { Plus, X, ArrowRight, Layers } from "lucide-react";
import clsx from "clsx";

export interface SpreadLeg {
    id: string;
    asset: string;
    side: "Buy" | "Sell";
    ratio: number;
}

export interface ComboBuilderProps {
    markets: { id: string; symbol: string; price: number }[];
    onSpreadChange: (legs: SpreadLeg[]) => void;
}

const PRESETS = [
    {
        name: "Calendar Spread",
        legs: [
            { side: "Buy", ratio: 1, assetOffset: 0 },
            { side: "Sell", ratio: 1, assetOffset: 1 }, // Simplistic mock for now
        ],
    },
    {
        name: "Butterfly",
        legs: [
            { side: "Buy", ratio: 1, assetOffset: 0 },
            { side: "Sell", ratio: 2, assetOffset: 1 },
            { side: "Buy", ratio: 1, assetOffset: 2 },
        ],
    },
    {
        name: "Straddle",
        legs: [
            { side: "Buy", ratio: 1, assetOffset: 0 }, // Call
            { side: "Buy", ratio: 1, assetOffset: 0 }, // Put (Needs different instrument type in reality)
        ],
    },
];

export function ComboBuilder({ markets, onSpreadChange }: ComboBuilderProps) {
    const [legs, setLegs] = useState<SpreadLeg[]>([
        { id: "1", asset: markets[0]?.symbol || "BTC", side: "Buy", ratio: 1 },
        { id: "2", asset: markets[0]?.symbol || "BTC", side: "Sell", ratio: 1 },
    ]);

    const updateLeg = (id: string, updates: Partial<SpreadLeg>) => {
        const newLegs = legs.map((leg) => (leg.id === id ? { ...leg, ...updates } : leg));
        setLegs(newLegs);
        onSpreadChange(newLegs);
    };

    const addLeg = () => {
        const newLeg: SpreadLeg = {
            id: Math.random().toString(36).substr(2, 9),
            asset: markets[0]?.symbol || "BTC",
            side: "Buy",
            ratio: 1,
        };
        const newLegs = [...legs, newLeg];
        setLegs(newLegs);
        onSpreadChange(newLegs);
    };

    const removeLeg = (id: string) => {
        const newLegs = legs.filter((l) => l.id !== id);
        setLegs(newLegs);
        onSpreadChange(newLegs);
    };

    const applyPreset = (presetName: string) => {
        // Mock implementation for presets
        // In a real app we'd need logic to select appropriate assets (e.g. different expiries)
        if (presetName === "Calendar Spread") {
            setLegs([
                { id: "p1", asset: markets[0]?.symbol || "BTC", side: "Buy", ratio: 1 },
                { id: "p2", asset: markets[1]?.symbol || markets[0]?.symbol || "ETH", side: "Sell", ratio: 1 },
            ]);
        }
    };

    // Calculate Net Price (Mock)
    const netPrice = legs.reduce((acc, leg) => {
        const market = markets.find(m => m.symbol === leg.asset);
        const price = market ? market.price : 0;
        return acc + (leg.side === "Buy" ? price * leg.ratio : -price * leg.ratio);
    }, 0);

    return (
        <div className="flex flex-col h-full bg-slate-900">
            {/* Presets Toolbar */}
            <div className="p-2 border-b border-slate-800 flex gap-2 overflow-x-auto">
                {PRESETS.map((p) => (
                    <button
                        key={p.name}
                        onClick={() => applyPreset(p.name)}
                        className="px-2 py-1 text-xs bg-slate-800 hover:bg-slate-700 text-slate-300 rounded border border-slate-700 whitespace-nowrap"
                    >
                        {p.name}
                    </button>
                ))}
            </div>

            {/* Legs List */}
            <div className="flex-1 overflow-y-auto p-2 space-y-2">
                {legs.map((leg, index) => (
                    <div key={leg.id} className="bg-slate-800/50 p-2 rounded border border-slate-700 flex items-center gap-2 text-sm">
                        <div className="w-6 text-center text-slate-500 text-xs font-mono">{index + 1}</div>

                        <select
                            value={leg.side}
                            onChange={(e) => updateLeg(leg.id, { side: e.target.value as "Buy" | "Sell" })}
                            className={clsx(
                                "bg-slate-900 border border-slate-700 rounded px-1 py-1 text-xs font-bold",
                                leg.side === "Buy" ? "text-emerald-400" : "text-rose-400"
                            )}
                        >
                            <option value="Buy">Buy</option>
                            <option value="Sell">Sell</option>
                        </select>

                        <input
                            type="number"
                            value={leg.ratio}
                            onChange={(e) => updateLeg(leg.id, { ratio: parseFloat(e.target.value) })}
                            className="w-12 bg-slate-900 border border-slate-700 rounded px-1 py-1 text-right text-xs"
                        />

                        <select
                            value={leg.asset}
                            onChange={(e) => updateLeg(leg.id, { asset: e.target.value })}
                            className="flex-1 bg-slate-900 border border-slate-700 rounded px-1 py-1 text-xs text-white"
                        >
                            {markets.map(m => (
                                <option key={m.id} value={m.symbol}>{m.symbol}</option>
                            ))}
                        </select>

                        <button onClick={() => removeLeg(leg.id)} className="text-slate-500 hover:text-rose-400">
                            <X size={14} />
                        </button>
                    </div>
                ))}

                <button onClick={addLeg} className="w-full py-2 border border-dashed border-slate-700 text-slate-500 hover:text-indigo-400 hover:border-indigo-500/50 rounded flex items-center justify-center gap-1 text-xs transition-colors">
                    <Plus size={12} /> Add Leg
                </button>
            </div>

            {/* Payoff / Net Price */}
            <div className="p-3 bg-slate-950 border-t border-slate-800 text-xs">
                <div className="flex justify-between items-center mb-2">
                    <span className="text-slate-500">Net Price</span>
                    <span className="font-mono text-white text-sm">${Math.abs(netPrice).toFixed(2)} {netPrice > 0 ? "Debit" : "Credit"}</span>
                </div>
                <div className="flex items-center gap-2 text-slate-500">
                    <Layers size={12} />
                    <span>{legs.length} Legs</span>
                </div>
            </div>
        </div>
    );
}
