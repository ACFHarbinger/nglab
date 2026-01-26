
import { useState } from "react";
import { Plus, Play, Save, Code } from "lucide-react";
import { nanoid } from "nanoid";
import { ConditionBlock, Condition } from "./ConditionBlock";
import { ActionBlock, StrategyAction } from "./ActionBlock";

interface Strategy {
    id: string;
    name: string;
    conditions: Condition[];
    actions: StrategyAction[];
}

export function StrategyBuilder() {
    const [name, setName] = useState("New Strategy");
    const [conditions, setConditions] = useState<Condition[]>([]);
    const [actions, setActions] = useState<StrategyAction[]>([]);

    const addCondition = () => {
        setConditions([
            ...conditions,
            { id: nanoid(), type: "Indicator", indicator: "RSI", operator: ">", value: 70, period: 14 }
        ]);
    };

    const addAction = () => {
        setActions([
            ...actions,
            { id: nanoid(), type: "Buy", quantity: 100, orderType: "Market" }
        ]);
    };

    const updateCondition = (id: string, newC: Condition) => {
        setConditions(conditions.map(c => c.id === id ? newC : c));
    };

    const updateAction = (id: string, newA: StrategyAction) => {
        setActions(actions.map(a => a.id === id ? newA : a));
    };

    return (
        <div className="flex flex-col h-full bg-slate-950 p-6 overflow-y-auto">
            <header className="flex justify-between items-center mb-6">
                <div>
                    <h2 className="text-2xl font-bold text-slate-100 mb-1">Strategy Builder</h2>
                    <p className="text-slate-400 text-sm">Design automated trading strategies visually.</p>
                </div>
                <div className="flex gap-2">
                    <button className="flex items-center gap-2 px-3 py-2 bg-slate-800 hover:bg-slate-700 rounded text-sm text-slate-200 transition-colors">
                        <Code size={16} /> View Code
                    </button>
                    <button className="flex items-center gap-2 px-3 py-2 bg-indigo-600 hover:bg-indigo-500 rounded text-sm font-bold text-white transition-colors shadow-lg shadow-indigo-500/20">
                        <Save size={16} /> Save Strategy
                    </button>
                </div>
            </header>

            <div className="flex flex-col gap-6 max-w-4xl mx-auto w-full">
                {/* Strategy Name */}
                <div className="bg-slate-900 rounded-xl p-4 border border-slate-800">
                    <label className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-2 block">Strategy Name</label>
                    <input
                        type="text"
                        value={name}
                        onChange={(e) => setName(e.target.value)}
                        className="bg-transparent text-xl font-bold text-white outline-none w-full placeholder:text-slate-600"
                        placeholder="Enter strategy name..."
                    />
                </div>

                {/* Logic Flow */}
                <div className="relative pl-8 border-l-2 border-slate-800 space-y-8">

                    {/* WHEN Block */}
                    <div className="relative">
                        <div className="absolute -left-[41px] top-0 bg-slate-800 text-slate-400 text-[10px] font-bold px-2 py-1 rounded border border-slate-700">WHEN</div>

                        <div className="space-y-3">
                            {conditions.map(c => (
                                <ConditionBlock
                                    key={c.id}
                                    condition={c}
                                    onChange={(nc) => updateCondition(c.id, nc)}
                                    onRemove={() => setConditions(conditions.filter(x => x.id !== c.id))}
                                />
                            ))}

                            <button
                                onClick={addCondition}
                                className="w-full py-3 border-2 border-dashed border-slate-800 rounded-lg text-slate-500 hover:border-slate-600 hover:text-slate-300 transition-all flex items-center justify-center gap-2 text-sm font-medium"
                            >
                                <Plus size={16} /> Add Condition
                            </button>
                        </div>
                    </div>

                    {/* THEN Block */}
                    <div className="relative">
                        <div className="absolute -left-[41px] top-0 bg-indigo-900/50 text-indigo-300 text-[10px] font-bold px-2 py-1 rounded border border-indigo-500/30">THEN</div>

                        <div className="space-y-3">
                            {actions.map(a => (
                                <ActionBlock
                                    key={a.id}
                                    action={a}
                                    onChange={(na) => updateAction(a.id, na)}
                                    onRemove={() => setActions(actions.filter(x => x.id !== a.id))}
                                />
                            ))}

                            <button
                                onClick={addAction}
                                className="w-full py-3 border-2 border-dashed border-indigo-900/30 rounded-lg text-indigo-400/50 hover:border-indigo-500/50 hover:text-indigo-300 transition-all flex items-center justify-center gap-2 text-sm font-medium"
                            >
                                <Plus size={16} /> Add Action
                            </button>
                        </div>
                    </div>

                </div>
            </div>
        </div>
    );
}
