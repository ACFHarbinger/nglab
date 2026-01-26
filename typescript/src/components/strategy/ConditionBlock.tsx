
import { useState } from "react";
import { Copy, Trash, X } from "lucide-react";

export interface Condition {
    id: string;
    type: "Price" | "Indicator" | "Time";
    indicator?: string; // e.g., "RSI", "SMA"
    operator: ">" | "<" | "=" | "Crosses Above" | "Crosses Below";
    value: number;
    period?: number; // for indicators
}

interface ConditionBlockProps {
    condition: Condition;
    onChange: (c: Condition) => void;
    onRemove: () => void;
}

export function ConditionBlock({ condition, onChange, onRemove }: ConditionBlockProps) {
    return (
        <div className="bg-slate-800 rounded p-3 border border-slate-700 flex flex-col gap-2">
            <div className="flex justify-between items-center">
                <span className="text-xs font-bold text-slate-400 uppercase tracking-widest">
                    {condition.type === "Price" ? "Price Check" : condition.type === "Time" ? "Time Check" : "Indicator Check"}
                </span>
                <button onClick={onRemove} className="text-slate-500 hover:text-rose-400">
                    <Trash size={14} />
                </button>
            </div>

            <div className="flex gap-2 items-center">
                <select
                    value={condition.type}
                    onChange={(e) => onChange({ ...condition, type: e.target.value as any })}
                    className="bg-slate-900 border border-slate-700 rounded text-xs px-2 py-1 text-slate-300 focus:border-indigo-500 outline-none"
                >
                    <option value="Price">Price</option>
                    <option value="Indicator">Indicator</option>
                    {/* <option value="Time">Time</option> */}
                </select>

                {condition.type === "Indicator" && (
                    <select
                        value={condition.indicator}
                        onChange={(e) => onChange({ ...condition, indicator: e.target.value })}
                        className="bg-slate-900 border border-slate-700 rounded text-xs px-2 py-1 text-slate-300 focus:border-indigo-500 outline-none"
                    >
                        <option value="RSI">RSI</option>
                        <option value="SMA">SMA</option>
                        <option value="EMA">EMA</option>
                    </select>
                )}

                <select
                    value={condition.operator}
                    onChange={(e) => onChange({ ...condition, operator: e.target.value as any })}
                    className="bg-slate-900 border border-slate-700 rounded text-xs px-2 py-1 text-slate-300 focus:border-indigo-500 outline-none w-32"
                >
                    <option value=">">Greater Than</option>
                    <option value="<">Less Than</option>
                    <option value="=">Equals</option>
                    <option value="Crosses Above">Crosses Above</option>
                    <option value="Crosses Below">Crosses Below</option>
                </select>

                <input
                    type="number"
                    value={condition.value}
                    onChange={(e) => onChange({ ...condition, value: parseFloat(e.target.value) })}
                    className="bg-slate-900 border border-slate-700 rounded text-xs px-2 py-1 text-slate-300 focus:border-indigo-500 outline-none w-20"
                />

                {condition.type === "Indicator" && (
                    <div className="flex items-center gap-1">
                        <span className="text-[10px] text-slate-500 uppercase">Period</span>
                        <input
                            type="number"
                            value={condition.period || 14}
                            onChange={(e) => onChange({ ...condition, period: parseFloat(e.target.value) })}
                            className="bg-slate-900 border border-slate-700 rounded text-xs px-2 py-1 text-slate-300 focus:border-indigo-500 outline-none w-12"
                        />
                    </div>
                )}
            </div>
        </div>
    );
}
