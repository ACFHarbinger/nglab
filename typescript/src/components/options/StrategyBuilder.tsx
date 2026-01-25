import React, { useState } from 'react';
import { Plus, Trash2, TrendingUp } from 'lucide-react';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: ClassValue[]) {
    return twMerge(clsx(inputs));
}

type LegType = 'Call' | 'Put';
type Action = 'Buy' | 'Sell';

interface StrategyLeg {
    id: string;
    type: LegType;
    action: Action;
    strike: number;
    expiry: string;
    quantity: number;
}

export const StrategyBuilder: React.FC = () => {
    const [legs, setLegs] = useState<StrategyLeg[]>([]);
    const [underlyingPrice, setUnderlyingPrice] = useState(100);

    const addLeg = () => {
        setLegs([...legs, {
            id: Math.random().toString(36).substr(2, 9),
            type: 'Call',
            action: 'Buy',
            strike: underlyingPrice,
            expiry: '2026-06-01',
            quantity: 1
        }]);
    };

    const updateLeg = (id: string, updates: Partial<StrategyLeg>) => {
        setLegs(legs.map(l => l.id === id ? { ...l, ...updates } : l));
    };

    const removeLeg = (id: string) => {
        setLegs(legs.filter(l => l.id !== id));
    };

    // Simple payoff calculation for visualization (at expiry)
    const calculatePayoff = (price: number) => {
        return legs.reduce((total, leg) => {
            let value = 0;
            if (leg.type === 'Call') {
                value = Math.max(0, price - leg.strike);
            } else {
                value = Math.max(0, leg.strike - price);
            }
            // Simplified: ignoring premium for now, just intrinsic value shape
            // In real app, we need option premiums.
            const direction = leg.action === 'Buy' ? 1 : -1;
            return total + (value * direction * leg.quantity);
        }, 0);
    };

    // Generate diagram points
    const points = [];
    const minStrike = Math.min(...legs.map(l => l.strike), underlyingPrice) * 0.8;
    const maxStrike = Math.max(...legs.map(l => l.strike), underlyingPrice) * 1.2;
    const step = (maxStrike - minStrike) / 50;

    for (let p = minStrike; p <= maxStrike; p += step) {
        points.push({ price: p, payoff: calculatePayoff(p) });
    }

    return (
        <div className="flex flex-col h-full bg-slate-900 text-slate-200 p-4 rounded-lg">
            <h2 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
                <TrendingUp size={24} className="text-blue-400" />
                Strategy Builder
            </h2>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 h-full">
                {/* Legs Configuration */}
                <div className="col-span-1 flex flex-col gap-4 bg-slate-800/50 p-4 rounded-lg">
                    <div className="flex justify-between items-center pb-2 border-b border-slate-700">
                        <h3 className="font-semibold text-slate-300">Legs</h3>
                        <button
                            onClick={addLeg}
                            className="flex items-center gap-1 bg-blue-600 hover:bg-blue-500 text-white px-2 py-1 rounded text-sm transition"
                        >
                            <Plus size={16} /> Add Leg
                        </button>
                    </div>

                    <div className="flex-1 overflow-auto flex flex-col gap-2">
                        {legs.length === 0 && (
                            <div className="text-center text-slate-500 py-8 italic">No legs added. Start building!</div>
                        )}
                        {legs.map(leg => (
                            <div key={leg.id} className="bg-slate-800 p-3 rounded border border-slate-700 flex flex-col gap-2">
                                <div className="flex gap-2">
                                    <select
                                        className={cn("rounded px-2 py-1 text-sm font-bold bg-slate-900 border",
                                            leg.action === 'Buy' ? "text-green-400 border-green-900" : "text-red-400 border-red-900"
                                        )}
                                        value={leg.action}
                                        onChange={(e) => updateLeg(leg.id, { action: e.target.value as Action })}
                                    >
                                        <option value="Buy">Buy</option>
                                        <option value="Sell">Sell</option>
                                    </select>
                                    <input
                                        type="number"
                                        className="w-16 bg-slate-900 border border-slate-700 rounded px-2 py-1 text-sm text-center"
                                        value={leg.quantity}
                                        onChange={(e) => updateLeg(leg.id, { quantity: parseInt(e.target.value) })}
                                    />
                                    <select
                                        className="rounded px-2 py-1 text-sm bg-slate-900 border border-slate-700"
                                        value={leg.type}
                                        onChange={(e) => updateLeg(leg.id, { type: e.target.value as LegType })}
                                    >
                                        <option value="Call">Call</option>
                                        <option value="Put">Put</option>
                                    </select>
                                    <button
                                        onClick={() => removeLeg(leg.id)}
                                        className="ml-auto text-slate-500 hover:text-red-400"
                                    >
                                        <Trash2 size={16} />
                                    </button>
                                </div>
                                <div className="flex gap-2 items-center text-sm">
                                    <span className="text-slate-400 w-12">Strike:</span>
                                    <input
                                        type="number"
                                        className="flex-1 bg-slate-900 border border-slate-700 rounded px-2 py-1 text-right"
                                        value={leg.strike}
                                        onChange={(e) => updateLeg(leg.id, { strike: parseFloat(e.target.value) })}
                                    />
                                </div>
                            </div>
                        ))}
                    </div>
                </div>

                {/* Payoff Diagram Visualization */}
                <div className="col-span-2 bg-slate-800/50 p-4 rounded-lg flex flex-col">
                    <h3 className="font-semibold text-slate-300 mb-4">Payout Profile (Intrinsic)</h3>
                    <div className="flex-1 bg-slate-900 rounded relative overflow-hidden flex items-end justify-center p-4">
                        {/* Simple SVG Chart */}
                        {points.length > 1 && maxStrike > minStrike && (
                            <svg className="w-full h-full" viewBox={`${minStrike} ${Math.min(...points.map(p => p.payoff))} ${maxStrike - minStrike} ${Math.max(...points.map(p => p.payoff)) - Math.min(...points.map(p => p.payoff)) + 10}`}>
                                {/* Needs proper scaling logic for real chart, simplified here */}
                                <path
                                    d={`M ${points.map(p => {
                                        // Normalize x and y for simple rendering if needed
                                        // This is placeholder logic as SVG path construction requires generic viewbox handling
                                        return `${p.price} ${-p.payoff}`;
                                    }).join(' L ')}`}
                                    fill="none"
                                    stroke="#3b82f6"
                                    strokeWidth="2"
                                />
                            </svg>
                        )}
                        <div className="absolute inset-0 flex items-center justify-center text-slate-500">
                            {/* Placeholder for Recharts or Lightweight Charts */}
                            [Payoff Diagram Visualization Placeholder]
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};
